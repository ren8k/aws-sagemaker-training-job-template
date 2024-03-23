import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pytz
import yaml


def load_config(config_path):
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)


def get_timestamp():
    # 日本時間をyyyymmddhhmmss形式で返す
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d-%H-%M-%S")


def save_json(data: dict, save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


def upload_to_s3(local_path: Path | str, s3_uri: str) -> None:
    local_path = Path(local_path)
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).upload_file(str(local_path), key)


def download_from_s3(uri: str, save_path: Path | str) -> None:
    save_path = Path(save_path)
    make_dir(save_path.parent)

    bucket, key = uri.replace("s3://", "").split("/", 1)
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, str(save_path))


def make_dir(path: Path | str) -> None:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def get_cloudwatch_logs():
    logs = boto3.client("logs")
    log_group_name = "/aws/sagemaker/TrainingJobs"
    latest_logstream_name = logs.describe_log_streams(
        logGroupName=log_group_name, orderBy="LastEventTime", descending=True
    )["logStreams"][0]["logStreamName"]
    log = logs.get_log_events(
        logGroupName=log_group_name,
        logStreamName=latest_logstream_name,
    )
    return log


def save_formatted_logs(logs: dict, save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # save "timestamp" and "message" to log file
    body = logs["events"]
    with open(save_path, "w") as f:
        for line in body:
            time = int(str(line["timestamp"])[:10])
            # convert UTC to JST
            dt_utc = datetime.fromtimestamp(time, pytz.utc)
            dt_jst = dt_utc.astimezone(pytz.timezone("Asia/Tokyo"))
            message = "[{}] {}\n".format(dt_jst, line["message"])
            f.write(message)
