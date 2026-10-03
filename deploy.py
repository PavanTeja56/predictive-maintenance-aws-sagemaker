import os
import boto3
from dotenv import load_dotenv
from sagemaker.serve import ModelBuilder
from sagemaker.core import image_uris

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE_ARN")

PIPELINE_NAME = "PredictiveMaintenancePipeline"
ENDPOINT_NAME = "predictive-maintenance-endpoint"

session = boto3.Session(region_name=AWS_REGION)
sagemaker_client = session.client("sagemaker")


# ---------------------------------------------------------
# 1. Get latest successful pipeline execution
# ---------------------------------------------------------

print("Finding latest pipeline execution...")

executions = sagemaker_client.list_pipeline_executions(
    PipelineName=PIPELINE_NAME,
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=10
)

successful_execution = None

for execution in executions["PipelineExecutionSummaries"]:
    status = execution["PipelineExecutionStatus"]

    print(
        f"Execution: {execution['PipelineExecutionArn']} | "
        f"Status: {status}"
    )

    if status == "Succeeded":
        successful_execution = execution
        break

if successful_execution is None:
    raise RuntimeError(
        "No successful pipeline execution was found."
    )

execution_arn = successful_execution["PipelineExecutionArn"]

print("\nUsing successful pipeline execution:")
print(execution_arn)


# ---------------------------------------------------------
# 2. Find TrainModel step
# ---------------------------------------------------------

print("\nFinding TrainModel step...")

steps = sagemaker_client.list_pipeline_execution_steps(
    PipelineExecutionArn=execution_arn
)

training_step = None

for step in steps["PipelineExecutionSteps"]:

    print(
        f"{step['StepName']} -> "
        f"{step['StepStatus']}"
    )

    if step["StepName"] == "TrainModel":
        training_step = step

if training_step is None:
    raise RuntimeError(
        "TrainModel step was not found."
    )


# ---------------------------------------------------------
# 3. Get training job
# ---------------------------------------------------------

training_metadata = training_step.get("Metadata", {})

if "TrainingJob" not in training_metadata:
    raise RuntimeError(
        "Training job metadata was not found in TrainModel step."
    )

training_job_arn = training_metadata["TrainingJob"]["Arn"]

training_job_name = training_job_arn.split("/")[-1]

print("\nTraining job:")
print(training_job_name)


# ---------------------------------------------------------
# 4. Get model artifact
# ---------------------------------------------------------

training_job = sagemaker_client.describe_training_job(
    TrainingJobName=training_job_name
)

model_artifact = training_job["ModelArtifacts"]["S3ModelArtifacts"]

print("\nModel artifact:")
print(model_artifact)


# ---------------------------------------------------------
# 5. Get SageMaker inference image
# ---------------------------------------------------------

print("\nGetting SageMaker inference image...")

inference_image = image_uris.retrieve(
    framework="sklearn",
    region=AWS_REGION,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.t3.medium",
    image_scope="inference"
)

print("Inference image:")
print(inference_image)


# ---------------------------------------------------------
# 6. Build SageMaker model
# ---------------------------------------------------------

print("\nBuilding SageMaker model...")

model_builder = ModelBuilder(
    model=model_artifact,
    role_arn=SAGEMAKER_ROLE,
    image_uri=inference_image,
    instance_type="ml.t3.medium"
)

model = model_builder.build(
    model_name="predictive-maintenance-model"
)

print("SageMaker model built successfully!")


# ---------------------------------------------------------
# 7. Deploy endpoint
# ---------------------------------------------------------

print("\nDeploying endpoint...")

endpoint = model.deploy(
    endpoint_name=ENDPOINT_NAME,
    initial_instance_count=1,
    instance_type="ml.t3.medium",
    wait=True
)

print("\n========================================")
print("Endpoint deployed successfully!")
print("========================================")
print(f"Endpoint name: {ENDPOINT_NAME}")