from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)


def get_timestamp():
    # 日本時間をyyyymmddhhmmss形式で返す
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d-%H-%M-%S")


def log(data: dict, save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        df: pd.DataFrame = pd.read_csv(save_path)
        df = pd.DataFrame(df.to_dict("records") + [data])
        df.to_csv(save_path, index=False)
    else:
        pd.DataFrame([data]).to_csv(save_path, index=False)


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
