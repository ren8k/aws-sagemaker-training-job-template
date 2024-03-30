# aws-sagemaker-training-job-template <!-- omit in toc -->

本リポジトリでは，SageMaker Training Job を利用した機械学習コードの実行と実験管理を容易に行うためのテンプレートとその利用手順を示す．テンプレート内では，[MNIST データを題材としたサンプルコード](https://github.com/Renya-Kujirada/aws-sagemaker-training-job-template/blob/main/src/train.py)を用意しており，ローカルで実行している機械学習コードを最小限の修正で SageMaker Training Job として実行し，実験管理を行うための方法を解説している．併せて，[サンプルコードを SageMaker Studio などでクイックに実行するための手順書](https://github.com/Renya-Kujirada/aws-sagemaker-training-job-template/blob/main/docs)も整備しているため参照されたい．

本テンプレートは，筆者が実務や Kaggle などでの SageMaker の利用経験を基に作成しており，ローカル，または SageMaker Training Job のどちらでも同一のコードで動作可能な状態にする方法についても述べている．

## TL;DR <!-- omit in toc -->

SageMaker Training Job を実行可能なコードテンプレートを Python コードベースで作成した．本テンプレートを利用することで，以下の機能を迅速かつ容易に利用することができる．

- SageMaker Training Job の実行（オプション指定可能）
- SageMaker Experiments を利用した実験管理
- Training Job 実行後，モデルおよびログの自動ダウンロード

## 目次 <!-- omit in toc -->

- [背景と課題](#背景と課題)
- [目的](#目的)
- [オリジナリティ](#オリジナリティ)
- [前提](#前提)
  - [環境](#環境)
  - [ディレクトリ構成](#ディレクトリ構成)
  - [コード](#コード)
- [手順](#手順)
- [手順の各ステップの詳細](#手順の各ステップの詳細)
  - [データセットの準備と S3 へのアップロード](#データセットの準備と-s3-へのアップロード)
  - [学習スクリプト（`train.py`）および依存関係ファイルを用意](#学習スクリプトtrainpyおよび依存関係ファイルを用意)
    - [データセットを格納しているディレクトリの設定](#データセットを格納しているディレクトリの設定)
    - [アーティファクト（モデル，メトリクス等）の保存先の設定](#アーティファクトモデルメトリクス等の保存先の設定)
    - [SageMaker Experiments の利用設定（任意）](#sagemaker-experiments-の利用設定任意)
  - [ローカル上での動作確認](#ローカル上での動作確認)
  - [ハイパーパラメーターを定義した yaml ファイルを`config`ディレクトリに格納](#ハイパーパラメーターを定義した-yaml-ファイルをconfigディレクトリに格納)
  - [Training Job を実行し，作成されたモデル・CloudWatch Logs を自動ダウンロード](#training-job-を実行し作成されたモデルcloudwatch-logs-を自動ダウンロード)
    - [`run_job.py`の概要](#run_jobpyの概要)
    - [実行方法](#実行方法)
    - [Training Job の実行結果の保存先](#training-job-の実行結果の保存先)
- [Tips](#tips)

## 背景と課題

Amazon SageMaker Training Job とは，① 用意したコードを ② 用意したデータと ③ 用意した環境で実行し，④ 結果を自動で保存するバッチ処理サービスである[^1-1]．SageMaker Training Job を利用することで，データサイエンティストは学習に必要なインフラ管理から開放され，機械学習のコード開発に注力することができる．また，SageMaker Training Job では，SageMaker Experiments という機能を利用することで，WandB や MLflow のような実験管理が容易に実現可能になる．通常，機械学習コードの開発初期はローカルで動作確認などを行い，ハイパーパラメーターチューニングや複数設定での比較実験などで SageMaker Training Job, Experiments を利用するケースが多い．

初学者にとって，ローカルで実行していた学習コードを SageMaker Training Job で動作するように修正することは，少々難しいように思われる．その理由としては，実務や Kaggle などの機械学習プロジェクトで即時転用可能な，Training Job の**Python コードベース**の実装例が少ないことが挙げられる．（AWS 公式リポジトリでは，Jupyter Notebook での解説コードは豊富に存在する．[^1-2] [^1-3] [^1-4]）加え，SageMaker Training Job 中で，SageMaker Experiments による実験管理を行うための実装例は非常に少ない．

## 目的

業務における機械学習の PoC や Kaggle において，ローカル上で開発した学習コードを迅速かつ容易に SageMaker Training Job として実行可能な Python コードを作成し，再利用可能なようにテンプレートとして整備する．また，テンプレート内に具体的な実装例を含め，SageMaker Training Job や SageMaker Experiments の利用方法を解説することも目的としている．本テンプレートを利用することにより，初学者が SageMaker 上での学習・実験管理を行えるようになることを狙いとしている．

## オリジナリティ

- ローカル・SageMaker Training Job 問わず，修正無しに実行可能な学習スクリプト(`train.py`)の実装例を紹介している
  - SageMaker Experiments の利用方法も紹介している
- Training Job を実行するための Python コード(`scripts/run_job.py`)を作成している
  - `train.py` のハイパーパラメータを外部の yaml ファイルで管理し，それを読み込み training job に渡すように工夫している
- SageMaker Training Job 実行後に以下を自動ダウンロードしている．
  - 学習済みモデル
  - CloudWatch Logs の実行ログ（失敗時には原因究明がスムーズになる）

## 前提

### 環境

- SageMaker Studio，または，sagemaker>=2.213.0 が install された ML 実行環境上での実行を想定している．
  - 本リポジトリは，AWS Deep Learning Containers Images をベースとした VSCode Dev Containers 上で開発を行っている．Training Job と同一環境での Training コードの動作確認を行えるため，開発効率が良い．詳細は[VSCode Dev Containers を利用した AWS EC2 上での開発環境構築手順](https://github.com/Renya-Kujirada/aws-ec2-devkit-vscode)を参照されたい．
- 機械学習フレームワークとして PyTorch の利用を想定している．
  - 勿論，TensorFlow，MXNet，HuggingFace などにも対応させることも可能．（`scripts/run_job.py` を修正する必要あり）

### ディレクトリ構成

本リポジトリは，以下のディレクトリ構成を想定している．`src`ディレクトリに学習スクリプト（`train.py`）を格納し，`scripts/run_job.py`によって`train.py`を SageMaker Training Job で実行する想定である．また，ローカル上で`train.py`を実行する際のデータセットの格納先は，`dataset`ディレクトリを想定している．

```
.
├── config      :   学習スクリプトのハイパーパラメーターを定義したyamlを格納
├── dataset     :   ローカルで学習スクリプトを実行する際に利用するデータセットを格納
├── rawdata     :   Rawデータを格納（任意）
├── result
│   ├── model   :   Training Job実行後にモデルを自動ダウンロード（格納）
│   └── output  :   ローカルで学習スクリプトを実行する際に利用するデータ保存先
├── scripts     :   Training Job実行用のスクリプトを格納
└── src         :   学習スクリプトを格納
```

### コード

- 以下のファイルは`src`ディレクトリに格納する
  - 学習スクリプト（`train.py`という名前を想定している）
  - `train.py`で利用しているモジュール
  - `train.py`の実行に必要な依存関係ファイル（`requirements.txt`）
    - Training Job 実行時，コンテナ上に自動で install される
- `train.py`内部では，`argparse`を利用してハイパーパラメーターを動的に変更できるようにする
  - SageMaker Experiments でメトリクスと紐付けて自動記録するため
- `train.py`で設定するハイパーパラメーターは，`config`ディレクトリ内部の yaml ファイルで管理する

## 手順

以下に，本テンプレートを利用して，ローカルで動作確認を行った ML コードをシームレスに SageMaker Training Job で実行するための手順を示す．

- データセットの準備と S3 へのアップロード
- 学習スクリプト（`train.py`）および依存関係ファイルを用意
- ハイパーパラメーターを定義した yaml ファイルを`config`ディレクトリに格納
- Training Job を実行し，作成されたモデル・CloudWatch Logs を自動ダウンロード

## 手順の各ステップの詳細

### データセットの準備と S3 へのアップロード

`dataset`ディレクトリに，`train.py`で利用するデータセットを準備する．その後，`src`ディレクトリ内部で以下のように`upload_dataset.py`を実行することで，データセットを S3 に upload する．

```
python upload_dataset.py
```

デフォルトのデータセット upload 先（S3 URI）は以下である．

```
s3://sagemaker-{REGION}-{ACCOUNT_ID}/dataset
```

なお，`upload_dataset.py`では，コマンドライン引数を指定することで，upload 先の S3 URI や region を変更可能である．（例えば，引数`--prefix`のデフォルト値は`dataset`となっているが，これを`dataset/pj-name`とすると，`s3://sagemaker-{REGION}-{ACCOUNT_ID}/dataset/pj-name`に upload されるようになる．）

### 学習スクリプト（`train.py`）および依存関係ファイルを用意

`train.py`，`train.py`で利用しているモジュール，および`train.py`の実行に必要な依存関係ファイル（`requirements.txt`）を`src`ディレクトリに格納する．参考のために，本リポジトリでは mnsit の画像分類のための`train.py`を作成している．

SageMaker Training Job で`train.py`を実行するために留意すべき点は以下である．

- データセットを格納しているディレクトリの設定
- アーティファクト（モデル，メトリクス等）の保存先の設定
- SageMaker Experiments の利用設定（任意）
- ローカル上での動作確認

以下，具体的な修正点を簡易解説する．

#### データセットを格納しているディレクトリの設定

`train.py`上では，`argparse`を利用して，データセット格納先を以下のように定義することを推奨する．

```py
parser.add_argument(
    "--data-dir",
    type=str,
    default=os.environ["SM_CHANNEL_TRAINING"],
)
```

Training Job が実行されるコンテナでは，指定した S3 上のデータセットが`/opt/ml/input/data/training`に転送され，コンテナ上の環境変数`SM_CHANNEL_TRAINING`にディレクトリパスが格納される仕様である．よって，コード上では，`args.data_dir`でデータセットのディレクトリパスにアクセスする．なお，Training Job では，他にも様々な環境変数が利用可能である[^2][^3]ので，実装の際には公式リポジトリなどを参考にされたい．

#### アーティファクト（モデル，メトリクス等）の保存先の設定

`train.py`上では，`argparse`を利用して，モデルやその他出力物の保存先を以下のように定義することを推奨する．

```py
parser.add_argument(
    "--model-dir",
    type=str,
    default=os.environ["SM_MODEL_DIR"],
)
parser.add_argument(
    "--out-dir",
    type=str,
    default=os.environ["SM_OUTPUT_DATA_DIR"],
)
```

前述の`SM_CHANNEL_TRAINING`と同様に，コンテナ上の環境変数`SM_MODEL_DIR`，`SM_OUTPUT_DATA_DIR`には，それぞれ`/opt/ml/model`，`/opt/ml/output`が格納されており，Training Job 終了後に S3 に自動で保存される仕様である．前述のディレクトリ以外は，Training Job 終了時に全て削除されるため，Job 実行時に生成されるモデルの重みファイルは`/opt/ml/model`に，その他ファイルは`/opt/ml/output`に保存すると良い．

#### SageMaker Experiments の利用設定（任意）

SageMaker Experiments は，SageMaker の機能の一つであり，機械学習の実験を追跡，整理，比較するための機能を提供する．噛み砕いて説明すると，MLflow の AWS 版だと考えれば良く，`Experiment`という単位の中に，実行毎に`Run`という単位でパラメーター（loss，accuracy の推移や混同行列，ハイパーパラメーターなど）を記録することができる．

Training Job 中で利用する場合，`train.py`上では，以下のように Training Job 実行時に指定された`Experiment`と`Run`の情報を渡す必要がある．

```py

from sagemaker.experiments import load_run

with load_run(experiment_name=args.exp_name, run_name=args.run_name) as run:
    train(args, run)
```

基本的には，`train.py`上で，Experiments の API を呼ぶことでパラメーターを記録することができる．例えば，混同行列を記録したい場合は，

```py
run.log_confusion_matrix(target.cpu(), pred.cpu(), "Confusion-Matrix-Test-Data")
```

のように記述し，epoch 毎のパラメーター値を記録したい場合は，

```py
run.log_metric(name="test:accuracy", value=accuracy, step=epoch)
```

のように記述すると良い．詳細については，公式ドキュメント[^4][^5][^6]やブログ[^7]を参考にされたい．

なお，本リポジトリ上では，ローカル上でも SageMaker Training Job 上でも同一コードで動作させるために，ローカル実行の場合は明示的に`run = None`としており，run によって，API を実行するか否かを自動判定させている．

### ローカル上での動作確認

SageMaker Training Job を実行する前に，SageMaker Training Job を模して ローカルで動作確認を行うことは，実験効率の観点で重要である．Training Job を実行する際，Job 実行用のインスタンス・コンテナ起動時間などの待ち時間が発生するためである．以下のような shell を作成し，実際に実行してみることを推奨する（本リポジトリでは，`src`ディレクトリ内に`train.sh`という shell を用意している）．

```sh
#!/bin/bash
cd "$(dirname "$0")"

export SM_CHANNEL_TRAINING="../dataset"
export SM_OUTPUT_DATA_DIR="../result/output"
export SM_MODEL_DIR="../result/model"

python train.py
```

`bash train.sh`のように実行することで，`dataset`ディレクトリ上のデータセットを入力とし，`result/model`ディレクトリには学習後のモデルの重みファイルが，`result/output`ディレクトリにはその他ファイル（本リポジトリの`train.py`の場合，epoch 毎のメトリクスとモデルの重みファイル）が保存されることを確認できる．

### ハイパーパラメーターを定義した yaml ファイルを`config`ディレクトリに格納

`train.py`上で`argparse`で指定しているハイパーパラメーターを`exp_<3桁の実験番号>.yaml`という名前で保存しておく．Training Job 実行時に`yaml.safe_load`で dict 形式で load し，SageMaker Estimator に容易に渡せるためである．

### Training Job を実行し，作成されたモデル・CloudWatch Logs を自動ダウンロード

`scripts/run_job.py`を実行することで，Training Job を実行することができる．

#### `run_job.py`の概要

`scripts`ディレクトリ内部で`run_job.py`を実行することで，`src`ディレクトリ内の`train.py`が Training Job によって実行される．また，Training Job により作成されたモデル，実行ログ（CloudWatch Logs），実験情報（モデルの s3 uri, および job name）は自動ダウンロードされる．なお，Training Job の成否に関わらず，CloudWatch Logs のログはダウンロードするよう実装している．これにより，Training Job の実行に失敗した場合，迅速にエラー解析が可能になる．

#### 実行方法

`run_job.py`では，SageMaker Pytorch Estimator の一部の引数をコマンドライン引数として指定することができる．全てを説明しないが，利用頻度が高そうなものを紹介する．

- `--config`: Pytorch Estimator の引数`hp`に渡すためのハイパーパラメーターを定義した yaml ファイルパス
- `--dataset-uri`: データセットを格納している S3 URI
- `--exp-name`: Training Job の job 名の prefix，および SageMaker Experiments 名
- `--instance-type`: インスタンスタイプ（デフォルトは`ml.g4dn.xlarge`）
- `--input-mode`: データセットを Training Job 開始前にコンテナにダウンロードするか，Training Job 実行中にストリーミングで取得するかを指定可能．詳細は公式ブログ[^9]を参照されたい．
- `--use-spot`: スポットインスタンスを使用するかを指定可能．(デフォルトでは利用しない)

`run_job.py`を容易に実行するために`run_job.sh`を用意している．`run_job.sh`の 9 行目，10 行目，12 行目を編集することで，利用可能である．以下に`run_job.sh`の中身を示す．9 行目の変数`EXP_NAME`には任意の実験名を，10 行目の変数`ACCOUNT_ID`には自身の AWS アカウント ID を，12 行目の変数`DATASET_S3_URI`には Training Job に転送したいデータセットの S3 URI を指定する．なお，12 行目は`src/upload_dataset.py`実行時に引数`--prefix`を指定していない場合は変更不要である．

```sh
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
```

その後，`scripts`ディレクトリ内部にて，以下のコマンドで，3 桁の実験番号を引数として指定して`run_job.sh`を実行する．ここで，3 桁の実験番号は，`config`ディレクトリの`exp_<3桁の実験番号>.yaml`のファイル名の末尾の番号である．

```sh
bash run_job.sh 001
```

上記コマンドにより，`src/run_job.py`が実行され，Training Job を実行することができる．

#### Training Job の実行結果の保存先

Training Job 実行に伴い作成される SageMaker Experiments Run 名，S3 へのモデルの保存先（S3 URI），およびローカルへのダウンロード先（ディレクトリ）を以下に示す．

- SageMaker Experiments Run 名: `run-{yyyy-mm-dd-hh-mm-ss}`
- モデル保存先（S3 URI）: `s3://sagemaker-{REGION}-{ACCOUNT_ID}/dataset/result-training-job-{self.exp_name}`
- モデルダウンロード先（ローカル）: `../result/model/{yyyy-mm-dd-hh-mm-ss}`

## Tips

- Training Job では，`SageMaker managed warm pools`を利用する前提である．本機能は，Training Job を実行後，その際に使用したインスタンスを停止せずに保持しておき，待ち時間無く Training Job を再実行可能な機能である．Warm pool を使用する場合，インスタンスタイプごとに上限緩和申請が必要である．詳細は[^10]を参照されたい

- 同一名の Experiments に紐付けられる Run の総数は 50 である（SageMaker が自動作成したものを除く）[^20]．50 を超えると以下のエラーが発生するため，Experiments Name を変更する必要がある．

```
botocore.errorfactory.ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the AssociateTrialComponent operation: The account-level service limit 'Total number of trial components allowed in a single trial, excluding those automatically created by SageMaker' is 50 Trial Components, with current utilization of 0 Trial Components and a request delta of 51 Trial Components. Please use AWS Service Quotas to request an increase for this quota. If AWS Service Quotas is not available, contact AWS support to request an increase for this quota.
```

- `train.py`内での SageMaker Experiments の実装について，本リポジトリ上では ExperimentsName 並びに RunName をトレーニングジョブ内のスクリプトに明示的に指定することで，`run_job.py`上で作成した Run を利用して記録するようにしているが，公式の実装例[^6]でも問題なく記録することが可能．具体的な実装例は以下．

```py
import boto3
from sagemaker.session import Session

session = Session(boto3.session.Session(region_name="ap-northeast-1"))
with load_run(sagemaker_session=session) as run:
    train(args, run)
```

- 現時点（2024/03/25）では，`run_job.py`上で作成した Run 上に，`train.py`上のメトリクスを記録できない．

## References <!-- omit in toc -->

[^1-1]: [エンジニア目線で始める Amazon SageMaker Training ① 機械学習を使わないはじめての Training Job](https://qiita.com/kazuneet/items/795e561efce8c874d115)
[^1-2]: [sagemaker/sagemaker-experiments/pytorch_mnist/src/mnist_train.py](https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-experiments/pytorch_mnist/src/mnist_train.py)
[^1-3]: [sagemaker/sagemaker-training/tutorial/2_2_rewriting_traing_code_for_sagemaker_pytorch.ipynb](https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-training/tutorial/2_2_rewriting_traing_code_for_sagemaker_pytorch.ipynb)
[^1-4]: [sagemaker/sagemaker-experiments/pytorch_mnist/pytorch_mnist.ipynb](https://github.com/aws-samples/aws-ml-jp/blob/main/sagemaker/sagemaker-experiments/pytorch_mnist/pytorch_mnist.ipynb)
[^2]: [ENVIRONMENT_VARIABLES.md ](https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md)
[^3]: [SageMaker Training Toolkit - ENVIRONMENT_VARIABLES.md 日本語版](https://zenn.dev/kmotohas/articles/7bfe313eab01ea)
[^4]: [Amazon SageMaker Experiments > Experiments](https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html)
[^5]: [Next generation Amazon SageMaker Experiments – Organize, track, and compare your machine learning trainings at scale](https://aws.amazon.com/jp/blogs/machine-learning/next-generation-amazon-sagemaker-experiments-organize-track-and-compare-your-machine-learning-trainings-at-scale/)
[^6]: [Track an experiment while training a Pytorch model with a SageMaker Training Job](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-experiments/sagemaker_job_tracking/pytorch_script_mode_training_job.html)
[^7]: [新しくなった Amazon SageMaker Experiments で実験管理](https://qiita.com/mariohcat/items/9fde1b04c0ecf439d427)
[^8]: [Training APIs > Estimators](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)
[^9]: [Choose the best data source for your Amazon SageMaker training job](https://aws.amazon.com/jp/blogs/machine-learning/choose-the-best-data-source-for-your-amazon-sagemaker-training-job/)
[^10]: [Train Using SageMaker Managed Warm Pools](https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html)
[^20]: [Amazon SageMaker endpoints and quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)
