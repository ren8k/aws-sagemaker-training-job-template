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
TIMESTAMP = src.utils.get_timestamp()
EXP_NAME = "mnist"


class Experiment:
    def __init__(self, args) -> None:
        self.region = REGION
        os.environ["AWS_DEFAULT_REGION"] = self.region
        self.session = sagemaker.Session()
        self.role = sagemaker.get_execution_role()
        self.hp = src.utils.load_config(args.config)
        self.instance_type = INSTANCE_TYPE
        self.conf_name = os.path.basename(args.config).split(".")[0]

        self.exp_name = EXP_NAME
        self.job_name = f"{self.exp_name}-{self.conf_name}-{TIMESTAMP}"
        self.run_name = f"run-{TIMESTAMP}"

        # add sm exp settings to hyperparameters
        self.hp["experiment-name"] = self.exp_name
        self.hp["run-name"] = self.run_name

    def run(self):
        estimator = PyTorch(
            entry_point="train.py",
            source_dir="src",
            role=self.role,
            framework_version="2.0.0",
            py_version="py310",
            instance_count=1,
            instance_type=self.instance_type,
            hyperparameters=self.hp,
            base_job_name=self.job_name,
            environment={"AWS_DEFAULT_REGION": self.region},
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
    with Run(
        experiment_name=exp.exp_name,
        sagemaker_session=exp.session,
        run_name=exp.run_name,
    ) as run:
        exp.run()


if __name__ == "__main__":
    args = get_args()
    main(args)
