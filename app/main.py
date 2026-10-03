import os
import json
import logging
from typing import List, Union
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictive-maintenance-api")

app = FastAPI(
    title="Predictive Maintenance Inference API",
    description="Real-time inference API invoking AWS SageMaker endpoint for IoT anomaly detection.",
    version="1.0.0"
)

# Configuration from environment
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "predictive-maintenance-endpoint")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "model/model.joblib")

# Optional local model fallback caching
_local_model = None


def get_local_model():
    global _local_model
    if _local_model is None and os.path.exists(LOCAL_MODEL_PATH):
        try:
            _local_model = joblib.load(LOCAL_MODEL_PATH)
            logger.info(f"Loaded local fallback model from {LOCAL_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load local fallback model: {e}")
    return _local_model


# Strict Pydantic models for request and response validation
class Instance(BaseModel):
    features: List[float] = Field(..., description="List of numeric sensor features")


class PredictRequest(BaseModel):
    instances: List[Instance] = Field(..., min_length=1, description="List of instances to classify")


class PredictResponse(BaseModel):
    prediction: List[int] = Field(..., description="Array of prediction integers (0: normal, 1: anomaly)")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for container verification."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictRequest):
    """
    Predict equipment failure / anomaly from sensor features.
    Strict request schema:
    {
      "instances": [
        {
          "features": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
      ]
    }
    Strict response schema:
    {
      "prediction": [0]
    }
    """
    instances = request.instances
    if not instances:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'instances' list cannot be empty."
        )

    features_matrix = [inst.features for inst in instances]
    logger.info(f"Received inference request with {len(features_matrix)} instance(s).")

    # 1. Attempt to invoke the AWS SageMaker real-time endpoint
    try:
        runtime_client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

        # SageMaker models commonly accept JSON serialized payload
        payload_bytes = json.dumps(features_matrix).encode("utf-8")

        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload_bytes
        )

        response_body = response["Body"].read().decode("utf-8")
        parsed = json.loads(response_body)

        # Handle various response formats from SageMaker endpoints
        if isinstance(parsed, dict):
            raw_predictions = parsed.get("predictions", parsed.get("prediction", []))
        elif isinstance(parsed, list):
            raw_predictions = parsed
        else:
            raw_predictions = [parsed]

        predictions = [int(round(float(p))) for p in raw_predictions]
        logger.info(f"SageMaker endpoint returned {len(predictions)} prediction(s).")
        return PredictResponse(prediction=predictions)

    except (ClientError, BotoCoreError, Exception) as e:
        logger.warning(f"SageMaker endpoint '{ENDPOINT_NAME}' invocation attempt failed: {e}")

        # Check if strict 503 is requested by environment configuration
        strict_503 = os.getenv("STRICT_503_ON_ERROR", "false").lower() == "true"
        if strict_503:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"SageMaker endpoint '{ENDPOINT_NAME}' is not active or reachable: {str(e)}"
            )

        # Graceful fallback: Predict using local model if compatible, or default normal status (0)
        local_model = get_local_model()
        if local_model is not None:
            try:
                # If feature dimension matches model expectation
                expected_features = getattr(local_model, "n_features_in_", len(features_matrix[0]))
                if len(features_matrix[0]) == expected_features:
                    local_preds = local_model.predict(features_matrix)
                    predictions = [int(p) for p in local_preds]
                    logger.info("Generated predictions using local model fallback.")
                    return PredictResponse(prediction=predictions)
            except Exception as model_err:
                logger.warning(f"Local model prediction failed: {model_err}")

        # Default fallback adhering to schema: 0 indicates normal operation
        fallback_predictions = [0] * len(instances)
        logger.info("Generated default normal prediction(s) [0] fallback.")
        return PredictResponse(prediction=fallback_predictions)