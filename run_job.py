import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session
from sagemaker.experiments.run import Run
import os

DATASET_S3_URI = "s3://sm-train-1710652103"

os.environ["AWS_DEFAULT_REGION"] = "ap-northeast-1"
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
instance_type = "ml.g4dn.xlarge"

experiment_name = "mnist-hand-written-digits-classification-2"

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    role=role,
    framework_version="2.0.0",
    py_version="py310",
    instance_count=1,
    instance_type=instance_type,
    hyperparameters={
        "batch-size": 128,
        "lr": 0.01,
        "epochs": 3,
        # "backend": "gloo",
    },
    # environment={"AWS_DEFAULT_REGION": "ap-nottheast-1"},
    # keep_alive_period_in_seconds=1800,
)
estimator.fit({"training": DATASET_S3_URI})
