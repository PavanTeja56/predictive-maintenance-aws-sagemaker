import os
import sys
import types
import boto3
from dotenv import load_dotenv

from sagemaker.core import image_uris
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingS3Input,
    ProcessingOutput,
    ProcessingS3Output
)
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import InputData, Compute, SourceCode
from sagemaker.mlops.workflow.steps import (
    ProcessingStep,
    TrainingStep
)
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.pipeline import Pipeline

# Provide backward-compatibility mapping for SageMaker workflow modules
if "sagemaker.workflow" not in sys.modules:
    sys.modules["sagemaker.workflow"] = types.ModuleType("sagemaker.workflow")
    sys.modules["sagemaker.workflow.pipeline"] = sys.modules["sagemaker.mlops.workflow.pipeline"]
    sys.modules["sagemaker.workflow.steps"] = sys.modules["sagemaker.mlops.workflow.steps"]

# Load environment variables
load_dotenv()

# AWS configuration
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "predictive-maintenance-2026-pavan-8472")
SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE_ARN")

# Create boto3 session
boto_session = boto3.Session(region_name=AWS_REGION)

# Create SageMaker Pipeline session
sagemaker_session = PipelineSession(boto_session=boto_session)

# -------------------------------------------------------------
# Step 1: Preprocessing Data (ProcessingStep)
# -------------------------------------------------------------
processing_image = image_uris.retrieve(
    framework="sklearn",
    region=AWS_REGION,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.t3.medium",
    image_scope="training"
)

sklearn_processor = ScriptProcessor(
    image_uri=processing_image,
    role=SAGEMAKER_ROLE,
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=sagemaker_session
)

preprocessing_step = ProcessingStep(
    name="PreprocessData",
    step_args=sklearn_processor.run(
        inputs=[
            ProcessingInput(
                input_name="raw-data",
                s3_input=ProcessingS3Input(
                    s3_uri=f"s3://{S3_BUCKET}/data/raw/ai4i2020.csv",
                    local_path="/opt/ml/processing/input",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="FullyReplicated"
                )
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed-data",
                s3_output=ProcessingS3Output(
                    s3_uri=f"s3://{S3_BUCKET}/data/processed",
                    local_path="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob"
                )
            )
        ],
        code="src/preprocess.py",
        arguments=[
            "--input",
            "/opt/ml/processing/input/ai4i2020.csv",
            "--output",
            "/opt/ml/processing/output"
        ],
        wait=False
    )
)

# -------------------------------------------------------------
# Step 2: Training Model (TrainingStep)
# -------------------------------------------------------------
training_image = image_uris.retrieve(
    framework="sklearn",
    region=AWS_REGION,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.large",
    image_scope="training"
)

training_source = SourceCode(
    source_dir="src",
    entry_script="train.py"
)

training_trainer = ModelTrainer(
    training_image=training_image,
    role=SAGEMAKER_ROLE,
    source_code=training_source,
    compute=Compute(
        instance_type="ml.m5.large",
        instance_count=1
    ),
    sagemaker_session=sagemaker_session,
    input_data_config=[
        InputData(
            channel_name="train",
            data_source=(
                preprocessing_step
                .properties
                .ProcessingOutputConfig
                .Outputs["processed-data"]
                .S3Output
                .S3Uri
            )
        )
    ]
)

training_step = TrainingStep(
    name="TrainModel",
    step_args=training_trainer.train(),
    depends_on=[preprocessing_step]
)

# -------------------------------------------------------------
# Step 3: Model Evaluation (ProcessingStep)
# -------------------------------------------------------------
evaluation_processor = ScriptProcessor(
    image_uri=processing_image,
    role=SAGEMAKER_ROLE,
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=sagemaker_session
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    step_args=evaluation_processor.run(
        inputs=[
            ProcessingInput(
                input_name="model",
                s3_input=ProcessingS3Input(
                    s3_uri=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    local_path="/opt/ml/processing/model",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="FullyReplicated"
                )
            ),
            ProcessingInput(
                input_name="test-data",
                s3_input=ProcessingS3Input(
                    s3_uri=(
                        preprocessing_step
                        .properties
                        .ProcessingOutputConfig
                        .Outputs["processed-data"]
                        .S3Output
                        .S3Uri
                    ),
                    local_path="/opt/ml/processing/test",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="FullyReplicated"
                )
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                s3_output=ProcessingS3Output(
                    s3_uri=f"s3://{S3_BUCKET}/evaluation",
                    local_path="/opt/ml/processing/evaluation",
                    s3_upload_mode="EndOfJob"
                )
            )
        ],
        code="src/evaluate.py",
        arguments=[
            "--model",
            "/opt/ml/processing/model/model.tar.gz",
            "--test",
            "/opt/ml/processing/test/test.csv",
            "--output",
            "/opt/ml/processing/evaluation"
        ],
        wait=False
    ),
    depends_on=[training_step]
)

# -------------------------------------------------------------
# Pipeline Definition
# -------------------------------------------------------------
pipeline = Pipeline(
    name="PredictiveMaintenancePipeline",
    steps=[
        preprocessing_step,
        training_step,
        evaluation_step
    ],
    sagemaker_session=sagemaker_session
)

if __name__ == "__main__":
    print("Pipeline defined successfully!")
    print("Steps:", [step.name for step in pipeline.steps])
    print("AWS Region:", AWS_REGION)
    print("S3 Bucket:", S3_BUCKET)
    print("SageMaker Role:", SAGEMAKER_ROLE)