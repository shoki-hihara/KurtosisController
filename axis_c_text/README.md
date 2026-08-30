# CTRL v5 — Axis C (テキストデータ実験)

CTRL v5 (kurtosis-based learning-rate controller) の汎用性検証のうち、
テキストデータ軸 (Penn Treebank / WikiText-2、正則化LSTM) のコード一式。

このリポジトリには**コード(学習スクリプト・コントローラ実装)のみ**を置いている。
実験結果・データ・論文原稿等は含まない。

## ファイル

- `train_text_lstm.py` — 学習スクリプト本体。CTRL / COSINE / ONECYCLE / PLATEAU /
  STEP / WARMUP_COSINE / GALA / CWGD の8手法を比較する。
- `run_pilot_text.sh` — pilot実行用シェルスクリプト。実行のたびに自分自身のいる
  ディレクトリ (= このリポジトリのclone先) を検出し、`git pull` で最新化してから
  `train_text_lstm.py` を呼び出す。
- `gala_optimizer.py` / `cwgd_controller.py` — GALA / CWGD ベースラインの実装
  (`train_text_lstm.py` が import する)。

## セットアップ (ラボサーバ、最初の1回だけ)

```bash
git clone <このリポジトリのURL> /data01/s_hihara/ctrl_v5_axis_c_text
cd /data01/s_hihara/ctrl_v5_axis_c_text
bash run_pilot_text.sh > pilot_log_text.txt 2>&1
cat pilot_log_text.txt
```

2回目以降も同じディレクトリで `bash run_pilot_text.sh` を実行するだけでよい。
スクリプトが起動時に自動で `git pull` するため、コードを更新した場合も
ファイルを手動でコピーし直す必要はない。

## データについて

PTB (`wojzaremba/lstm` の GitHub raw から自動取得) と WikiText-2
(`Salesforce/wikitext`, HuggingFace `datasets` 経由) はどちらも
`run_pilot_text.sh` 実行時に自動取得される。このリポジトリには含めない。
