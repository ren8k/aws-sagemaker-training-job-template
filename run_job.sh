#!/bin/bash
cd "$(dirname "$0")"

EXP_ID=$1 # three digits number for experiment id
CONF_PATH=/app/sm-train/config/exp$EXP_ID.yaml

python src/run_job.py --config $CONF_PATH
