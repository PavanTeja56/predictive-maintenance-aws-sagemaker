from pipelines.maintenance_pipeline import pipeline, SAGEMAKER_ROLE


print("Uploading pipeline to SageMaker...")

pipeline.upsert(
    role_arn=SAGEMAKER_ROLE
)

print("Starting pipeline execution...")

execution = pipeline.start()

print("Pipeline started successfully!")
print("Execution ARN:")
print(execution.arn)