#!/bin/bash
cd "$(dirname "$0")"
BUCKETNAME=sm-train-$(date +%s)
REGION=ap-northeast-1
DATA_DIR=../dataset

aws s3api create-bucket --bucket $BUCKETNAME --region $REGION \
    --create-bucket-configuration LocationConstraint=$REGION
aws s3 cp $DATA_DIR s3://$BUCKETNAME --recursive
