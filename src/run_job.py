import argparse
import os

import sagemaker
from sagemaker import image_uris
from sagemaker.experiments.run import Run
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session

import utils

TIMESTAMP = utils.get_timestamp()


class Experiment:
    def __init__(self, args) -> None:
        self.region = args.region
        os.environ["AWS_DEFAULT_REGION"] = self.region
        self.session = sagemaker.Session()
        self.dataset_uri = args.dataset_uri
        self.instance_type = args.instance_type
        self.entry_point = args.entry_point
        if args.use_spot:
            self.kwargs = {
                "use_spot_instances": True,
                "max_run": 3600,
                "max_wait": 7200,
            }
        else:
            # Spot training job can't retain cluster.
            self.kwargs = {"keep_alive_period_in_seconds": 1800}

        # load hyperparameters from config file and add sm exp settings
        self.hp = utils.load_config(args.config)
        self.exp_name = args.exp_name
        self.conf_name = os.path.basename(args.config).split(".")[0]
        self.job_name = f"{self.exp_name}-{self.conf_name}-{TIMESTAMP}"
        self.run_name = f"run-{TIMESTAMP}"

        # add sm exp settings to hyperparameters
        self.hp["exp-name"] = self.exp_name
        self.hp["run-name"] = self.run_name

    def _get_image_uri(self):
        return image_uris.retrieve(
            framework="pytorch",
            version="2.0.1",
            py_version="py310",
            image_scope="training",
            region=self.region,
            instance_type=self.instance_type,
        )

    def run(self):
        estimator = PyTorch(
            entry_point=self.entry_point,
            source_dir="src",
            role=sagemaker.get_execution_role(),
            image_uri=self._get_image_uri(),
            instance_count=1,
            instance_type=self.instance_type,
            hyperparameters=self.hp,
            base_job_name=self.job_name,
            environment={"AWS_DEFAULT_REGION": self.region},
            **self.kwargs,
        )

        estimator.fit({"training": self.dataset_uri})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config")
    parser.add_argument("--dataset-uri", type=str, required=True, help="Dataset S3 URI")
    parser.add_argument("--exp-name", type=str, default="exp", help="Experiment name")
    parser.add_argument(
        "--instance-type", type=str, default="ml.g4dn.xlarge", help="InstanceType"
    )
    parser.add_argument(
        "--region", type=str, default="ap-northeast-1", help="region name"
    )
    parser.add_argument(
        "--entry-point", type=str, default="train.py", help="entry point"
    )
    parser.add_argument("--use-spot", action="store_true", help="Use spot instances")
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
