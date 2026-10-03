# Predictive Maintenance Pipeline with AWS SageMaker for IoT Anomaly Detection

## Project Overview

The **Predictive Maintenance Pipeline** is an end-to-end cloud-native machine learning solution designed to predict equipment failure and anomalous machine behavior from IoT sensor telemetry data. Predictive maintenance is a critical application of artificial intelligence in industrial manufacturing, energy, and aerospace, helping prevent costly unplanned downtime, optimize maintenance scheduling, and avoid catastrophic equipment breakdown.

This repository implements a modular, reproducible, and automated MLOps workflow leveraging:
- **AWS SageMaker Pipelines**: Orchestrating data preprocessing, distributed training, and model evaluation within a Directed Acyclic Graph (DAG).
- **AWS SageMaker Processing Jobs**: Cleaning, feature engineering, and evaluating model metrics.
- **AWS SageMaker Training Jobs**: Training high-performance classification models (such as Random Forest and XGBoost) on preprocessed sensor metrics.
- **AWS SageMaker Real-Time Endpoints**: Serving low-latency, real-time predictions to client applications.
- **Containerized FastAPI Inference Service**: Exposing a containerized REST API with strict request/response validation and automated fallback capabilities.

---

## Architecture

The system follows an automated MLOps architecture where each phase is decoupled, versioned, and orchestrated:

```
+-----------------------------------------------------------------------------------+
|                                  AWS Cloud                                        |
|                                                                                   |
|   +-------------------+        +--------------------+        +----------------+   |
|   |   Raw Data S3     |        |   Processed Data   |        | Model Artifact |   |
|   |  (ai4i2020.csv)   |        |   train / test     |        | (model.tar.gz) |   |
|   +---------+---------+        +---------+----------+        +-------+--------+   |
|             |                            ^                           ^            |
|             v                            |                           |            |
|     [ Preprocessing ] -------------------+                           |            |
|   (src/preprocess.py)                    |                           |            |
|     ProcessingStep                       v                           |            |
|                                  [ Model Training ] -----------------+            |
|                                   (src/train.py)                     |            |
|                                    TrainingStep                      v            |
|                                                           [ Model Evaluation ]    |
|                                                           (src/evaluate.py)       |
|                                                             ProcessingStep        |
|                                                                      |            |
|                                                                      v            |
|                                                               (evaluation.json)   |
+----------------------------------------------------------------------|------------+
                                                                       |
                                                                       v
+-----------------------------------------------------------------------------------+
|                             Deployment & Serving                                  |
|                                                                                   |
|           deploy.py  ==========>  AWS SageMaker Real-Time Endpoint                |
|                                                    ^                              |
|                                                    | (invoke_endpoint)            |
|   Client Application / Docker App                  v                              |
|   +-------------------------------------------------------------------------+     |
|   |  FastAPI Server (app/main.py)                                           |     |
|   |  - GET  /health   -> Status verification                                |     |
|   |  - POST /predict  -> Strict schema validation & real-time inference     |     |
|   +-------------------------------------------------------------------------+     |
+-----------------------------------------------------------------------------------+
```

### Pipeline Workflow Components:
1. **Data Preprocessing (`src/preprocess.py`)**:
   - Ingests raw telemetry records (e.g., AI4I 2020 Predictive Maintenance dataset).
   - Drops non-predictive identifiers, encodes operational variants, and performs stratified 80/10/10 train-validation-test splitting.
   - Outputs `train.csv`, `test.csv`, and `validation.csv`.
2. **Model Training (`src/train.py`)**:
   - Consumes training data from `/opt/ml/input/data/`.
   - Trains a balanced Random Forest classification model.
   - Serializes trained model artifacts to `/opt/ml/model/`.
3. **Model Evaluation (`src/evaluate.py`)**:
   - Assesses model performance against the holdout test dataset.
   - Writes structured performance metrics (`accuracy`, `f1_score`) to `/opt/ml/processing/evaluation/evaluation.json`.
4. **Pipeline Orchestration (`pipelines/maintenance_pipeline.py`)**:
   - Assembles pipeline steps into a `sagemaker.workflow.pipeline.Pipeline` instance with explicit dependencies (`depends_on`).
5. **Real-Time Model Deployment (`deploy.py`)**:
   - Retrieves model artifacts from the latest successful pipeline run and deploys a SageMaker managed endpoint.
6. **Inference API Service (`app/main.py`)**:
   - Serves `POST /predict` and `GET /health` within a Docker container.

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- AWS Account with an IAM Role configured with permissions for Amazon SageMaker and Amazon S3

### 1. Environment Configuration
Create a `.env` file from the provided `.env.example` template:

```bash
cp .env.example .env
```

Edit `.env` with your AWS credentials, region, and target S3 bucket:

```ini
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-s3-bucket-name
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/your-sagemaker-execution-role
ENDPOINT_NAME=predictive-maintenance-endpoint
```

Configure your AWS CLI credentials locally if you have not already:
```bash
aws configure
```

### 2. Local Python Environment Setup
Create a virtual environment and install project dependencies:

```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Docker Environment Setup
The local execution environment is containerized using Docker Compose. Build and run the service:

```bash
docker-compose up --build -d
```

Verify that the application service is running healthy:
```bash
docker-compose ps
curl http://localhost:8000/health
```

---

## Usage

### 1. Execute the SageMaker Pipeline
To create or update the pipeline definition and trigger an execution run on AWS SageMaker, execute:

```bash
python run_pipeline.py
```

This script:
- Connects to your SageMaker session.
- Calls `.upsert()` to register or update `PredictiveMaintenancePipeline`.
- Calls `.start()` to launch asynchronous pipeline execution in the AWS cloud.

### 2. Deploy the Model Endpoint
Once the pipeline has completed and model artifacts are stored in S3, deploy the model to a managed SageMaker real-time endpoint:

```bash
python deploy.py
```

This script:
- Locates the latest successful execution of `PredictiveMaintenancePipeline`.
- Extracts the S3 path of the trained model artifact.
- Provisions a managed endpoint using the SageMaker SDK (`model.deploy()`).

### 3. Perform Real-Time Inference via REST API

Start the local API service:
```bash
docker-compose up -d
```

#### Health Check
```bash
curl -X GET http://localhost:8000/health
```
Response:
```json
{
  "status": "healthy"
}
```

#### POST /predict
Send sensor readings to the `/predict` endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "features": [0.1, 0.2, 0.3, 0.4, 0.5]
      }
    ]
  }'
```

Response:
```json
{
  "prediction": [0]
}
```
*Note: A prediction of `0` denotes normal equipment operation, while `1` denotes an anomaly / failure risk.*
