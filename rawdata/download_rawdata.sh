#!/bin/bash

cd "$(dirname "$0")"
aws s3 cp s3://fast-ai-imageclas/mnist_png.tgz . --no-sign-request
tar -zxvf mnist_png.tgz
rm mnist_png.tgz
