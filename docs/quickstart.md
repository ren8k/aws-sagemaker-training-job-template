# クイックスタート

本リポジトリのサンプルコードを動かす手順を解説する．

## ディレクトリ構成

ディレクトリ構成は以下のようになっている．

```
.
├── config         :    学習スクリプトのハイパーパラメーターを定義した設定ファイルを格納
├── dataset        :    `src/preprocess.py`によって前処理されたデータセットを格納
├── docs           :    ドキュメントを格納
├── rawdata        :    rawデータをダウンロードし格納
├── result         :    `src/train.py`の実行結果を格納
│   ├── model      :    モデルの重みを保存
│   └── output     :    モデルの学習時のその他出力物を保存
├── scripts        :    SageMaker Training Jobを実行するためのコードを格納
└── src            :    学習スクリプトを格納
```

## 環境準備

EC2 上で VSCode Dev Container を起動，もしくは sagemaker>=2.213.0 が install された ML 実行環境を構築する．特に，EC2 上で Dev Container を利用する手順は，[VSCode Dev Containers を利用した AWS EC2 上での開発環境構築手順](https://github.com/Renya-Kujirada/aws-ec2-devkit-vscode)を参照されたい．

## データセット準備

`rawdata`ディレクトリに移動し，以下のコマンドを実行する．

```sh
bash download.sh
```

## Raw データの前処理

`src`ディレクトリに移動し，以下のコマンドを実行する．

```sh
python preprocess.py
```

### 前処理したデータを S3 へアップロード

以下のコマンドを実行する．

```sh
python upload_dataset.py
```

### ローカルで`train.py`を動作確認

以下のコマンドを実行する．

```sh
bash train.sh
```

### Training Job の実行

`scripts`ディレクトリに移動し，`run_job.sh`の 10 行目に，自身の AWS アカウント ID を記載する．その後，以下のコマンドを実行する．

```sh
bash run_job.sh
```
