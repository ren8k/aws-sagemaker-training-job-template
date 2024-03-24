#!/bin/bash
cd "$(dirname "$0")"

## config setting
EXP_ID=$1 # three digits number for experiment id
CONF_PATH=../config/exp$EXP_ID.yaml

## experiments setting
EXP_NAME=mnist
ACCOUNT_ID=XXXXXXXXXXXX
REGION=ap-northeast-1
DATASET_S3_URI=s3://sagemaker-$REGION-$ACCOUNT_ID/dataset
INSTANCE_TYPE=ml.g4dn.xlarge
OUT_DIR="../result/model"

# if you use spot instance, add --use-spot
python run_job.py --config $CONF_PATH \
    --dataset-uri $DATASET_S3_URI \
    --exp-name $EXP_NAME \
    --instance-type $INSTANCE_TYPE \
    --region $REGION \
    --out-dir $OUT_DIR
