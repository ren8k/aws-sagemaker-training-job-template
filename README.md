# aws-sagemaker-training-job-template

mnist を題材に，train.py をローカル，および sagemaker 上で実行できるコードを紹介する．
併せて，実験管理も行えるようにする．

## オリジナリティ

- ローカルでも SageMaker Training Job 上でもコードの改修無しに実行できるようにしている
- sagemaker training job を実際に実行しやすく整備したコード例が少なかった
  - train.py の hp を外部 yaml で管理し，それを読み込み training job に渡すように工夫している
- sagemaker experimtents を実際に適用したコード例が少ない
  - ローカルでも問題なく実行可能なように記述している

## 写真を交えた解説

## モチベ

- 自分用の sagemaker training job 実行テンプレートを作成したかった
- sagemaker experiments のサンプルコードが少ない
  - 現時点（2024/03/17）では，ExperimentsName 並びに RunName をトレーニングジョブ内のスクリプトに明示的に指定する必要がある．（qiita を参考にした）
    - 以下の公式ドキュメント通りでもうまくいく
    - https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-experiments/sagemaker_job_tracking/pytorch_script_mode_training_job.html
- warm pool はいいぞ！！

```py
import boto3
from sagemaker.session import Session

session = Session(boto3.session.Session(region_name="ap-northeast-1"))
with load_run(sagemaker_session=session) as run:
    train(args, run)
```

## 目次

- train.py の実行
- train.py のリファクタリング
- train.py を sm で実行
- sm-experiments の実装

- warm pool の開放（申請すること）

## reference

### sagemaker experiments

- https://qiita.com/mariohcat/items/9fde1b04c0ecf439d427
- https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-experiments/sagemaker_job_tracking/pytorch_script_mode_training_job.html

### sagemaker training job

- https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-experiments/pytorch_mnist/src/mnist_train.py
- https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-training/tutorial/2_2_rewriting_traing_code_for_sagemaker_pytorch.ipynb
