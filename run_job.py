import sagemaker
import argparse
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session
from sagemaker.experiments.run import Run
import os
import src.utils


DATASET_S3_URI = "s3://sm-train-1710652103"
REGION = "ap-northeast-1"
INSTANCE_TYPE = "ml.g4dn.xlarge"
JOB_NAME = "mnist"


class Experiment:
    def __init__(self, args) -> None:
        os.environ["AWS_DEFAULT_REGION"] = REGION
        self.session = sagemaker.Session()
        self.role = sagemaker.get_execution_role()
        self.hp = src.utils.load_config(args.config)
        self.instance_type = INSTANCE_TYPE
        self.job_name = f"{JOB_NAME}-{os.path.basename(args.config).split('.')[0]}"

    def run(self):
        estimator = PyTorch(
            entry_point="train.py",
            source_dir="src",
            role=self.role,
            framework_version="2.0.0",
            py_version="py310",
            instance_count=1,
            instance_type=INSTANCE_TYPE,
            hyperparameters=self.hp,
            base_job_name=self.job_name,
            # environment={"AWS_DEFAULT_REGION": "ap-nottheast-1"},
            keep_alive_period_in_seconds=1800,
        )

        estimator.fit({"training": DATASET_S3_URI})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    return parser.parse_args()


def main(args):
    exp = Experiment(args)
    exp.run()


if __name__ == "__main__":
    args = get_args()
    main(args)
