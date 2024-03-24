import argparse
import os

import sagemaker
import utils
from sagemaker import image_uris
from sagemaker.experiments.run import Run
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class Experiment:
    def __init__(self, args) -> None:
        self.time_stamp = utils.get_timestamp()
        self.save_artifact_dir = os.path.join(args.out_dir, self.time_stamp)
        utils.make_dir(self.save_artifact_dir)

        self.region = args.region
        os.environ["AWS_DEFAULT_REGION"] = self.region
        self.session = sagemaker.Session()
        self.dataset_uri = args.dataset_uri
        self.instance_type = args.instance_type
        self.entry_point = args.entry_point
        self.src_dir = os.path.join(BASE_DIR, "..", args.src_dir)
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
            version="2.2.0",
            py_version="py310",
            image_scope="training",
            region=self.region,
            instance_type=self.instance_type,
        )

    def run(self):
        estimator = PyTorch(
            entry_point=self.entry_point,
            source_dir=self.src_dir,
            role=sagemaker.get_execution_role(),
            image_uri=self._get_image_uri(),
            instance_count=1,
            instance_type=self.instance_type,
            hyperparameters=self.hp,
            base_job_name=self.job_name,
            output_path=f"s3://{self.session.default_bucket()}/result-training-job-{self.exp_name}",
            environment={"AWS_DEFAULT_REGION": self.region},
            **self.kwargs,
        )

        estimator.fit(
            inputs={"training": self.dataset_uri},
            wait=True,
            job_name=self.job_name,
        )
        return estimator

    def save_model(self, estimator):
        model_uri = estimator.latest_training_job.describe()["ModelArtifacts"][
            "S3ModelArtifacts"
        ]
        save_path = os.path.join(self.save_artifact_dir, "model.tar.gz")
        utils.download_from_s3(model_uri, save_path)
        os.system(f"tar -zxvf {save_path} -C {self.save_artifact_dir}")
        return model_uri

    def save_cloudwatch_log(self):
        cloudwatch_log = utils.get_cloudwatch_logs()
        utils.save_formatted_logs(
            cloudwatch_log, os.path.join(self.save_artifact_dir, "log.log")
        )

    def save_exp_info(self, model_uri):
        log_data = {
            "model_uri": model_uri,
            "job_name": self.job_name,
        }
        utils.save_json(log_data, os.path.join(self.save_artifact_dir, "exp.json"))


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
    parser.add_argument("--src-dir", type=str, default="src", help="Source directory")
    parser.add_argument("--use-spot", action="store_true", help="Use spot instances")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(BASE_DIR, "../output/model"),
        help="Output directory",
    )
    return parser.parse_args()


def main(args):
    exp = Experiment(args)
    try:
        with Run(
            experiment_name=exp.exp_name,
            sagemaker_session=exp.session,
            run_name=exp.run_name,
        ) as run:
            estimator = exp.run()
            model_uri = exp.save_model(estimator)
            exp.save_cloudwatch_log()
            exp.save_exp_info(model_uri)
            print("Finish training job")
    except Exception as e:
        print(f"Error: {e}")
        exp.save_cloudwatch_log()
        raise e


if __name__ == "__main__":
    args = get_args()
    main(args)
