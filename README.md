# aws-sagemaker-training-job-template

## はじめに

mnist を題材に，train.pyをローカル，および sagemaker 上で実行できるコードを紹介する．
併せて，実験管理も行えるようにする．

MLOpsの文脈等で実験管理は利用されがちだが，PoCでも使いたい．

可能な限り，

## TL;DR


## 目次

- 背景と課題
- 目的・解決方法
- オリジナリティ
- 前提
- 手順
- 手順の各ステップの詳細


## 背景と課題

## 目的・解決方法

## オリジナリティ

- ローカルでも SageMaker Training Job 上でもコードの改修無しに実行できるようにしている
- sagemaker training job を実際に実行しやすく整備したコード例が少なかった
  - train.py の hp を外部 yaml で管理し，それを読み込み training job に渡すように工夫している
- sagemaker experimtents を実際に適用したコード例が少ない
  - ローカルでも問題なく実行可能なように記述している
- SageMaker Training Job実行後に学習済みモデルを自動取得するようにしている
- SageMaker Training Jobの実行ログを成功失敗問わず自動取得するようにしている
  - 失敗時には原因究明がスムーズになる

## 前提

- 以下のファイルは`src`ディレクトリに格納する
  - 学習スクリプト（`train.py`）
  - `train.py`で利用しているモジュール
  - `train.py`の実行に必要な依存関係ファイル（`requirements.txt`）
- `train.py`内部では，`argparse`を利用してハイパーパラメーターを動的に変更できるようにする
  - SageMaker Experimentsでメトリクスと紐付けて自動記録するため
- `train.py`で設定するハイパーパラメーターは，`config`ディレクトリ内部のyamlファイルで管理する


## 手順

- 学習スクリプト（`train.py`）および依存関係ファイルを用意
- データセットをS3にアップロード
- ハイパーパラメーターを定義したyamlファイルを`config`ディレクトリに格納
- Training Jobを実行

## 手順の各ステップの詳細

### 学習スクリプト（`train.py`）および依存関係ファイルを用意

`train.py`，`train.py`で利用しているモジュール，および`train.py`の実行に必要な依存関係ファイル（`requirements.txt`）を`src`ディレクトリに格納する．参考として，本リポジトリではmnsitの画像分類のための`train.py`を作成している．

SageMaker Training Jobで`train.py`を実行するために留意すべき点は以下である．

- 

### データセットをS3にアップロード

`dataset`ディレクトリに利用するデータセットを格納し，`upload_dataset.py`を実行することでS3にupload可能．upload先は以下．

`The S3 URI of the uploaded file(s): s3://sagemaker-ap-northeast-1-081978453918/dataset`


### ハイパーパラメーターを定義したyamlファイルを`config`ディレクトリに格納

Training Job実行時にloadすることで，SageMaker Estimatorに容易に渡せる

### Training Jobを実行

写真を交えた解説も行う．


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


- ちなみに，experiments name, run nameを指定しなかった場合でも，問題なく記録可能っぽい，，この仕様よくわからん．．

## 目次

- train.py の実行
- train.py のリファクタリング
- train.py を sm で実行
- sm-experiments の実装

- warm pool の開放（申請すること）

## 使い方

- s3 に学習データを upload
- `src` ディレクトリ内に実行したいコードを格納
  - コード名は`train.py`を想定している
  - 依存関係があるコードもまとめて格納
  - lib も追加で入れたければ requirements.txt に追記
- run_job.sh を実行
  - training jobの実行
  - モデルのダウンロードおよび，job_nameの記録も自動でやってくれる



## TODO
- sagemaker upload経由でデータセットをuploadする
- jobはs3のどこに保存されるか？
  - mnistディレクトリのタイムスタンプディレクトリに格納される



### 実行

spot instanceを利用したい場合：--use-spotを引数に追加
デフォルトではkeep_alive=30分となっている

## Tips

- 同一名のExperimentsに紐付けられるRunの総数は50である（SageMakerが自動作成したものを除く）[^01]．50を超えると以下のエラーが発生するため，Experiments Nameを変更する必要がある．

```
botocore.errorfactory.ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the AssociateTrialComponent operation: The account-level service limit 'Total number of trial components allowed in a single trial, excluding those automatically created by SageMaker' is 50 Trial Components, with current utilization of 0 Trial Components and a request delta of 51 Trial Components. Please use AWS Service Quotas to request an increase for this quota. If AWS Service Quotas is not available, contact AWS support to request an increase for this quota.
```



## reference

[^01]: [Amazon SageMaker endpoints and quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)

### sagemaker experiments

#### official

- [Track an experiment while training a Pytorch model with a SageMaker Training Job](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-experiments/sagemaker_job_tracking/pytorch_script_mode_training_job.html)
- [Next generation Amazon SageMaker Experiments – Organize, track, and compare your machine learning trainings at scale](https://aws.amazon.com/jp/blogs/machine-learning/next-generation-amazon-sagemaker-experiments-organize-track-and-compare-your-machine-learning-trainings-at-scale/)

#### blog

- [新しくなった Amazon SageMaker Experiments で実験管理](https://qiita.com/mariohcat/items/9fde1b04c0ecf439d427)

- [SageMaker Processing で前処理を行って Training で学習したモデルのパラメータや精度を Experiments で記録する](https://www.sambaiz.net/article/442/)

- https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-experiments/sagemaker_job_tracking/pytorch_script_mode_training_job.html

### sagemaker training job

#### official

- [sagemaker/sagemaker-experiments/pytorch_mnist/src/mnist_train.py](https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-experiments/pytorch_mnist/src/mnist_train.py)
- [sagemaker/sagemaker-training/tutorial/2_2_rewriting_traing_code_for_sagemaker_pytorch.ipynb](https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-training/tutorial/2_2_rewriting_traing_code_for_sagemaker_pytorch.ipynb)
- [sagemaker/sagemaker-experiments/pytorch_mnist/pytorch_mnist.ipynb](https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-experiments/pytorch_mnist/pytorch_mnist.ipynb)

#### blog

- [エンジニア目線で始める Amazon SageMaker Training ①機械学習を使わないはじめてのTraining Job](https://qiita.com/kazuneet/items/795e561efce8c874d115)
- [SageMaker で学習ジョブを実行する ~組み込みアルゴリズム~](https://nsakki55.hatenablog.com/entry/2022/05/30/235551)
- [Amazon SageMakerで独自アルゴリズムを使ったトレーニング(学習)の作り方](https://qiita.com/shirakiya/items/b43c190958331c9825d3)
- [SageMaker入門者向け - 資料リンク集 -](https://qiita.com/Roe/items/fecb88176f1d29e99e0b)
