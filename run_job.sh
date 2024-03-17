#!/bin/bash
EXP_ID=$1 # three digits number for experiment id
CONF_PATH=/app/sm-train/config/exp$EXP_ID.yaml

python run_job.py --config $CONF_PATH
