#!/bin/bash
cd "$(dirname "$0")"

export SM_CHANNEL_TRAINING="/app/sm-train/dataset"
export SM_OUTPUT_DATA_DIR="/app/sm-train/result/output"
export SM_MODEL_DIR="/app/sm-train/result/model"

python train.py
