# aws-sagemaker-training-job-template

mnistを題材に，train.pyをローカル，およびsagemaker上で実行できるコードを紹介する．
併せて，実験管理も行えるようにする．


## モチベ
- 自分用のsagemaker training job実行テンプレートを作成したかった
- sagemaker experimentsのサンプルコードが少ない
  - 現時点（2024/03/17）では，ExperimentsName並びにRunNameをトレーニングジョブ内のスクリプトに明示的に指定する必要がある．（qiitaを参考にした）
    - 以下の公式ドキュメント通りでもうまくいく
    - https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-experiments/sagemaker_job_tracking/pytorch_script_mode_training_job.html
- warm poolはいいぞ！！


```py
import boto3
from sagemaker.session import Session

session = Session(boto3.session.Session(region_name="ap-northeast-1"))
with load_run(sagemaker_session=session) as run:
    train(args, run)
```

## 目次

- train.pyの実行
- train.pyのリファクタリング
- train.pyをsmで実行
- sm-experimentsの実装


- warm poolの開放（申請すること）

## reference

- sagemaker experiments: https://qiita.com/mariohcat/items/9fde1b04c0ecf439d427
