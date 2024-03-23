import argparse
import os

import sagemaker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region", type=str, default="ap-northeast-1", help="Region name"
    )
    parser.add_argument("--prefix", type=str, default="dataset/mnist", help="S3 prefix")
    parser.add_argument(
        "--upload-dir", type=str, default="../dataset", help="Upload directory"
    )
    return parser.parse_args()


def upload_dataset(session, upload_dir, bucket, prefix):
    s3_uri = session.upload_data(path=upload_dir, bucket=bucket, key_prefix=prefix)
    print("The S3 URI of the uploaded file(s): {}".format(s3_uri))


def main(args):
    os.environ["AWS_DEFAULT_REGION"] = args.region
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    upload_dataset(sagemaker_session, args.upload_dir, bucket, args.prefix)


if __name__ == "__main__":
    args = get_args()
    main(args)
