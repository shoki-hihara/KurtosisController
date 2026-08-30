# CTRL v5 — Axis C (テキストデータ実験)

[shoki-hihara/KurtosisController](https://github.com/shoki-hihara/KurtosisController)
リポジトリの `axis_c_text/` サブディレクトリ。CTRL v5 (kurtosis-based
learning-rate controller) の汎用性検証のうち、テキストデータ軸
(Penn Treebank / WikiText-2、正則化LSTM) のコード一式。

このリポジトリ全体には**コード(学習スクリプト・コントローラ実装)のみ**を
置いている(実験結果・データ・論文原稿等は含まない)。将来的に全実験コード
(CIFAR-100本実験・Axis A画像・Axis B表データ・v6診断実験)をこのリポジトリに
axisごとのサブディレクトリで集約する方針(2026-08-30)。表データ軸のCTRL v6
診断実験は `v6_diagnostics/` を参照。

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
git clone https://github.com/shoki-hihara/KurtosisController.git \
    /data01/s_hihara/KurtosisController
cd /data01/s_hihara/KurtosisController/axis_c_text
bash run_pilot_text.sh > pilot_log_text.txt 2>&1
cat pilot_log_text.txt
```

他axis用に既にこのリポジトリをcloneしている場合は、そのディレクトリで
`git pull` した上で `axis_c_text/` に入るだけでよい。改めてcloneし直す
必要はない。

2回目以降は `<cloneしたディレクトリ>/axis_c_text` で `bash run_pilot_text.sh`
を実行するだけでよい。スクリプトが起動時に自動で `git pull` するため、
コードを更新した場合もファイルを手動でコピーし直す必要はない
(★2026-08-30: サブディレクトリ構成への移行に伴い、リポジトリ検出ロジックを
`git rev-parse --is-inside-work-tree` ベースに修正済み。旧版のまま
axis_c_text/ に置くと、この自動更新が黙って効かなくなる不具合があったため
要注意)。

## データについて

PTB (`wojzaremba/lstm` の GitHub raw から自動取得) と WikiText-2
(`Salesforce/wikitext`, HuggingFace `datasets` 経由) はどちらも
`run_pilot_text.sh` 実行時に自動取得される。このリポジトリには含めない。
