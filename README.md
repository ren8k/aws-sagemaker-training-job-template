# aws-sagemaker-training-job-template

## はじめに

Amazon SageMaker Training とは，①用意したコードを②用意したデータと③用意した環境で実行してくれ，④結果を自動で保存してくれる，バッチ処理サービスである．[^1]

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

- SageMaker Studio, SageMaker Python SDKがinstallされたML実行環境上での実行を想定している．
  - 本リポジトリは，AWS Deep Learning Containers Imagesをベースとした VSCode Dev Containers 上で開発を行っている．Training Jobと同一環境でのTrainingコードの動作確認を行えるため，開発効率が良い．詳細は[VSCode Dev Containers を利用した AWS EC2 上での開発環境構築手順](https://github.com/Renya-Kujirada/aws-ec2-devkit-vscode)を参照されたい．

- 機械学習フレームワークとしてPytorchの利用を想定している．
  - 勿論，TensorFlow，MXNet，HuggingFaceなどにも対応させることも可能．（run_job.pyを修正する必要あり）

- 以下のファイルは`src`ディレクトリに格納する
  - 学習スクリプト（`train.py`）
  - `train.py`で利用しているモジュール
  - `train.py`の実行に必要な依存関係ファイル（`requirements.txt`）
- `train.py`内部では，`argparse`を利用してハイパーパラメーターを動的に変更できるようにする
  - SageMaker Experimentsでメトリクスと紐付けて自動記録するため
- `train.py`で設定するハイパーパラメーターは，`config`ディレクトリ内部のyamlファイルで管理する

## 手順

以下の手順では，本テンプレートを利用して，ローカルで動作確認を行ったMLコードをシームレスにSageMaker Training Jobで実行するための手順である．

- データセットの準備とS3へのアップロード
- 学習スクリプト（`train.py`）および依存関係ファイルを用意
- ハイパーパラメーターを定義したyamlファイルを`config`ディレクトリに格納
- Training Jobを実行

## 手順の各ステップの詳細

### データセットの準備とS3へのアップロード

`dataset`ディレクトリに，`train.py`で利用するデータセットを準備する．その後，`src`ディレクトリ上で以下のように`upload_dataset.py`を実行することで，データセットをS3にuploadする．

```
python upload_dataset.py
```

デフォルトのデータセットupload先（S3 URI）は以下である．

```
s3://sagemaker-{REGION}-{ACCOUNT_ID}/dataset
```

なお，`upload_dataset.py`では，コマンドライン引数を指定することで，upload先のS3 URIやregionを変更可能である．（例えば，引数`--prefix`のデフォルト値は`dataset`となっているが，これを`dataset/pj-name`とすると，`s3://sagemaker-{REGION}-{ACCOUNT_ID}/dataset/pj-name`にuploadされるようになる．）

### 学習スクリプト（`train.py`）および依存関係ファイルを用意

`train.py`，`train.py`で利用しているモジュール，および`train.py`の実行に必要な依存関係ファイル（`requirements.txt`）を`src`ディレクトリに格納する．参考のために，本リポジトリではmnsitの画像分類のための`train.py`を作成している．

SageMaker Training Jobで`train.py`を実行するために留意すべき点は以下である．

- データセットを格納しているディレクトリの設定
- アーティファクト（モデル，メトリクス等）の保存先の設定
- SageMaker Experimentsの利用設定（任意）
- Local上での動作確認

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

Training Jobが実行されるコンテナでは，指定したS3上のデータセットが`/opt/ml/input/data/training`に転送され，コンテナ上の環境変数`SM_CHANNEL_TRAINING`にディレクトリパスが格納される仕様である．よって，コード上では，`args.data_dir`でデータセットのディレクトリパスにアクセスする．なお，Training Jobでは，他にも様々な環境変数が利用可能である[^2][^3]ので，実装の際には公式リポジトリなどを参考にされたい．

### アーティファクト（モデル，メトリクス等）の保存先の設定

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

前述の`SM_CHANNEL_TRAINING`と同様に，コンテナ上の環境変数`SM_MODEL_DIR`，`SM_OUTPUT_DATA_DIR`には，それぞれ`/opt/ml/model`，`/opt/ml/output`が格納されており，Training Job終了後にS3に自動で保存される仕様である．前述のディレクトリ以外は，Training Job終了時に全て削除されるため，Job実行時に生成されるモデルの重みファイルは`/opt/ml/model`に，その他ファイルは`/opt/ml/output`に保存すると良い．

### SageMaker Experimentsの利用設定（任意）

SageMaker Experimentsは，SageMakerの機能の一つであり，機械学習の実験を追跡，整理，比較するための機能を提供する．噛み砕いて説明すると，MLflowのAWS版だと考えれば良く，`Experiment`という単位の中に，実行毎に`Run`という単位でパラメーター（loss，accuracyの推移や混同行列，ハイパーパラメーターなど）を記録することができる．

Training Job中で利用する場合，`train.py`上では，以下のようにTraining Job実行時に指定された`Experiment`と`Run`の情報を渡す必要がある．

```py

from sagemaker.experiments import load_run

with load_run(experiment_name=args.exp_name, run_name=args.run_name) as run:
    train(args, run)
```

基本的には，`train.py`上で，ExperimentsのAPIを呼ぶことでパラメーターを記録することができる．例えば，混同行列を記録したい場合は，

```py
run.log_confusion_matrix(target.cpu(), pred.cpu(), "Confusion-Matrix-Test-Data")
```
のように記述し，epoch毎のパラメーター値を記録したい場合は，

```py
run.log_metric(name="test:accuracy", value=accuracy, step=epoch)
```
のように記述すると良い．詳細については，公式ドキュメント[^4][^5][^6]やブログ[^7]を参考にされたい．

なお，本リポジトリ上では，local上でもSageMaker Training Job上でも同一コードで動作させるために，local実行の場合は明示的に`run = None`としており，runによって，APIを実行するか否かを自動判定させている．

### Local上での動作確認

SageMaker Training Jobを実行する前に，SageMaker Training Jobを模したLocalでの動作確認を行うことは，実験効率の観点で重要である．Training Jobを実行する際，Job実行用のインスタンス・コンテナ起動時間などの待ち時間が発生するためである．以下のようなshellを作成し，実際に実行してみることを推奨する（本リポジトリでは，`train.sh`というshellを用意している）．

```sh
#!/bin/bash
cd "$(dirname "$0")"

export SM_CHANNEL_TRAINING="../dataset"
export SM_OUTPUT_DATA_DIR="../result/output"
export SM_MODEL_DIR="../result/model"

python train.py

```

`bash train.sh`のように実行することで，`dataset`ディレクトリ上のデータセットを入力とし，`result/model`ディレクトリには学習後のモデルの重みファイルが，`result/output`ディレクトリにはその他ファイルが保存されることを確認することができる．

### ハイパーパラメーターを定義したyamlファイルを`config`ディレクトリに格納

`train.py`上で`argparse`で指定しているハイパーパラメーターを`exp_<3桁の実験番号>.yaml`という名前で保存しておく．Training Job実行時に`yaml.safe_load`でdict形式でloadし，SageMaker Estimatorに容易に渡せるためである．

### Training Jobを実行



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

---


写真を交えた解説も行う．


- 自分用の sagemaker training job 実行テンプレートを作成したかった
- sagemaker experiments のサンプルコードが少ない
  - 現時点（2024/03/17）では，ExperimentsName 並びに RunName をトレーニングジョブ内のスクリプトに明示的に指定する必要がある．（qiita を参考にした）
    - 以下の公式ドキュメント通りでもうまくいく
    - https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-experiments/sagemaker_job_tracking/pytorch_script_mode_training_job.html
- warm pool はいいぞ！！
- その他，FastFile modeなどもあるので，適宜利用されたい．

- s3://{SageMakerのdefault bucket}/{job_name}/model/mode.tar.gzに配置されます。また、SageMakerのdefault bucket は sagemaker-{REGION}-{ACCOUNT_ID}

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

- 同一名のExperimentsに紐付けられるRunの総数は50である（SageMakerが自動作成したものを除く）[^10]．50を超えると以下のエラーが発生するため，Experiments Nameを変更する必要がある．

```
botocore.errorfactory.ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the AssociateTrialComponent operation: The account-level service limit 'Total number of trial components allowed in a single trial, excluding those automatically created by SageMaker' is 50 Trial Components, with current utilization of 0 Trial Components and a request delta of 51 Trial Components. Please use AWS Service Quotas to request an increase for this quota. If AWS Service Quotas is not available, contact AWS support to request an increase for this quota.
```



## reference

[^1]: [エンジニア目線で始める Amazon SageMaker Training ①機械学習を使わないはじめてのTraining Job](https://qiita.com/kazuneet/items/795e561efce8c874d115)

[^2]: [ENVIRONMENT_VARIABLES.md ](https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md)

[^3]: [SageMaker Training Toolkit - ENVIRONMENT_VARIABLES.md 日本語版](https://zenn.dev/kmotohas/articles/7bfe313eab01ea)

[^4]: [Amazon SageMaker Experiments](https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html)

[^5]: [Next generation Amazon SageMaker Experiments – Organize, track, and compare your machine learning trainings at scale](https://aws.amazon.com/jp/blogs/machine-learning/next-generation-amazon-sagemaker-experiments-organize-track-and-compare-your-machine-learning-trainings-at-scale/)

[^6]: [Track an experiment while training a Pytorch model with a SageMaker Training Job](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-experiments/sagemaker_job_tracking/pytorch_script_mode_training_job.html)

[^7]: [新しくなった Amazon SageMaker Experiments で実験管理](https://qiita.com/mariohcat/items/9fde1b04c0ecf439d427)

[^10]: [Amazon SageMaker endpoints and quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)

### sagemaker experiments

#### official




#### blog

- 

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
