#!/bin/bash
cd "$(dirname "$0")"

EXP_ID=$1 # three digits number for experiment id

CONF_PATH=/app/sm-train/config/exp$EXP_ID.yaml
DATASET_S3_URI=s3://sm-train-1710652103
EXP_NAME=mnist
INSTANCE_TYPE=ml.g4dn.xlarge
REGION=ap-northeast-1
OUT_DIR="./output/model"

python src/run_job.py --config $CONF_PATH \
    --dataset-uri $DATASET_S3_URI \
    --exp-name $EXP_NAME \
    --instance-type $INSTANCE_TYPE \
    --region $REGION \
    --out-dir $OUT_DIR
