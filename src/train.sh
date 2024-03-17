#!/bin/bash
cd "$(dirname "$0")"
export SM_CHANNEL_TRAINING="/app/sm-train/dataset"
export SM_OUTPUT_DATA_DIR="/app/sm-train/output/metrics"
export SM_MODEL_DIR="/app/sm-train/output/model"

python train.py
