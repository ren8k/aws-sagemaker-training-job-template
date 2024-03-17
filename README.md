# aws-sagemaker-training-job-template

mnistを題材に，train.pyをローカル，およびsagemaker上で実行できるコードを紹介する．
併せて，実験管理も行えるようにする．


## オリジナリティ

- ローカルでもSageMaker Training Job上でもコードの改修無しに実行できるようにしている
- sagemaker training jobを実際に実行しやすく整備したコード例が少なかった
  - train.pyのhpを外部yamlで管理し，それを読み込みtraining jobに渡すように工夫している
- sagemaker experimtentsを実際に適用したコード例が少ない
  - ローカルでも問題なく実行可能なように記述している

## 写真を交えた解説

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

### sagemaker experiments
- https://qiita.com/mariohcat/items/9fde1b04c0ecf439d427


### sagemaker training job
- https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-experiments/pytorch_mnist/src/mnist_train.py
- https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-training/tutorial/2_2_rewriting_traing_code_for_sagemaker_pytorch.ipynb
