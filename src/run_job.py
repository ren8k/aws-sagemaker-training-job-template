import argparse
import os

import sagemaker
from sagemaker import image_uris
from sagemaker.experiments.run import Run
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session

import utils

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class Experiment:
    def __init__(self, args) -> None:
        self.time_stamp = utils.get_timestamp()
        self.region = args.region
        os.environ["AWS_DEFAULT_REGION"] = self.region
        self.session = sagemaker.Session()
        self.dataset_uri = args.dataset_uri
        self.instance_type = args.instance_type
        self.entry_point = args.entry_point
        if args.use_spot:
            self.kwargs = {
                "use_spot_instances": True,
                "max_run": 14400,
                "max_wait": 14400,
            }
        else:
            # Spot training job can't retain cluster.
            self.kwargs = {"keep_alive_period_in_seconds": 1800}

        # load hyperparameters from config file and add sm exp settings
        self.hp = utils.load_config(args.config)
        self.exp_name = args.exp_name
        self.conf_name = os.path.basename(args.config).split(".")[0]
        self.job_name = f"{self.exp_name}-{self.conf_name}-{self.time_stamp}"
        self.run_name = f"run-{self.time_stamp}"

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
            source_dir=BASE_DIR,
            role=sagemaker.get_execution_role(),
            image_uri=self._get_image_uri(),
            instance_count=1,
            instance_type=self.instance_type,
            hyperparameters=self.hp,
            base_job_name=self.job_name,
            output_path=f"s3://{self.session.default_bucket()}/{self.exp_name}",
            environment={"AWS_DEFAULT_REGION": self.region},
            **self.kwargs,
        )

        estimator.fit(
            inputs={"training": self.dataset_uri},
            wait=True,
            job_name=self.job_name,
        )
        return estimator

    def download_artifact(self, estimator, out_dir):
        save_dir = os.path.join(out_dir, self.time_stamp)
        utils.make_dir(save_dir)

        # save model
        model_uri = estimator.latest_training_job.describe()["ModelArtifacts"][
            "S3ModelArtifacts"
        ]
        save_path = os.path.join(save_dir, "model.tar.gz")
        utils.download_from_s3(model_uri, save_path)
        os.system(f"tar -zxvf {save_path} -C {save_dir}")

        # save experiment info
        log_data = {
            "model_uri": model_uri,
            "job_name": self.job_name,
        }
        utils.save_json(log_data, os.path.join(save_dir, "exp.json"))

        # save cloudwatch logs
        cloudwatch_log = utils.get_cloudwatch_logs()
        # utils.save_json(cloudwatch_log, os.path.join(save_dir, "log.json"))
        utils.save_formatted_logs(cloudwatch_log, os.path.join(save_dir, "log.log"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config")
    parser.add_argument("--dataset-uri", type=str, required=True, help="Dataset S3 URI")
    parser.add_argument("--exp-name", type=str, default="exp", help="Experiment name")
    parser.add_argument(
        "--instance-type", type=str, default="ml.g4dn.xlarge", help="InstanceType"
    )
    parser.add_argument(
        "--region", type=str, default="ap-northeast-1", help="Region name"
    )
    parser.add_argument(
        "--entry-point", type=str, default="train.py", help="Entry point file name"
    )
    parser.add_argument("--use-spot", action="store_true", help="Use spot instances")
    parser.add_argument(
        "--out-dir", type=str, default="./output/model", help="Output directory"
    )
    return parser.parse_args()


def main(args):
    exp = Experiment(args)
    with Run(
        experiment_name=exp.exp_name,
        sagemaker_session=exp.session,
        run_name=exp.run_name,
    ) as run:
        estimator = exp.run()
        exp.download_artifact(estimator, args.out_dir)  # option
        print("Finish training job")


# def test():
#     os.environ["AWS_DEFAULT_REGION"] = "ap-northeast-1"
#     # save cloudwatch logs
#     save_dir = "/app/sm-train/hoge"
#     cloudwatch_log = utils.get_cloudwatch_logs()
#     # utils.save_json(cloudwatch_log, os.path.join(save_dir, "log.json"))
#     utils.save_formatted_logs(cloudwatch_log, os.path.join(save_dir, "log.log"))


if __name__ == "__main__":
    args = get_args()
    main(args)
