# CTRL v6 診断実験 — 表データ (FT-Transformer) + SGD

このドキュメントは、[shoki-hihara/KurtosisController](https://github.com/shoki-hihara/KurtosisController)
リポジトリの `v6_diagnostics/` サブディレクトリに関するもの。このリポジトリは
CIFAR-100本実験・Axis A(画像汎用性)・Axis B(表データ汎用性)・Axis C(テキスト
汎用性、`train_text_lstm.py`)などをaxisごとのサブディレクトリで管理する構成
(将来的に全実験コードをこのリポジトリに集約する方針のため)。`gala_optimizer.py` /
`cwgd_controller.py` はこのサブディレクトリ用のコピー(他axisのサブディレクトリにも
同一内容のコピーが置かれる想定。内容は完全に同一)。

## 背景

CTRL v5 の汎用性検証 (Axis B、表データ) で、CTRL が AdamW+FT-Transformer では
ほぼ完全に no-op (COSINE と区別がつかない) になる現象が確認された。この現象の
原因として、「CTRL は SGD 系の訓練ダイナミクスを前提に設計されており、Adam 系
optimizer では機能しにくいのではないか」という仮説が挙がっている。

この診断実験は、その仮説を確認するための**小規模な検証専用**のコード。
FT-Transformer のアーキテクチャはそのままに、optimizer backbone だけを
AdamW → SGD に差し替えて、CTRL の介入挙動 (kurtosis の推移・intervention 回数) が
変化するかを見る。

**Axis B 本実験 (`train_tabular_ft_transformer.py`、このリポジトリには含まれない)
とは完全に別物であり、ここでの結果は本実験の結果とは混同しない。** CTRL 自体
(`ContinuousStateKurtosisController`)・`CTRL_CONFIG` は一切変更していない。

## このドキュメントに関連するファイル

- `train_tabular_sgd_diag.py` — `train_tabular_ft_transformer.py` (2026-08-29時点)
  を複製し、`--optimizer_override sgd` オプションのみを追加したもの。未指定 (既定)
  なら元のスクリプトと全く同じ挙動 (AdamW)。差分は `SGD_DIAG_TABULAR_CONFIG`
  (`lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0`。表データ軸で既に
  GALA baseline が使っている `GALA_TABULAR_CONFIG` と同じ値を流用) と、それを
  optimizer 構築部分・CLI 引数に配線しただけ。
- `run_sgd_diag_tabular.sh` — 実行用シェルスクリプト。`run_pilot_text.sh` と
  同じく、自分自身のいるディレクトリ (= このリポジトリの clone 先) を検出して
  `git pull` してから実行する。`smoke` (seed 0 のみ) と `full` (seed 0/1/2) の
  2モード。

## セットアップ (ラボサーバ)

このリポジトリを他axis用に既に `git clone` 済みであれば、そのディレクトリで
`git pull` した上で `v6_diagnostics/` に入るだけでよい。まだの場合:

```bash
git clone https://github.com/shoki-hihara/KurtosisController.git /data01/s_hihara/KurtosisController
cd /data01/s_hihara/KurtosisController/v6_diagnostics
bash run_sgd_diag_tabular.sh smoke > sgd_diag_smoke_log.txt 2>&1
cat sgd_diag_smoke_log.txt
```

2回目以降は `<cloneしたディレクトリ>/v6_diagnostics` で
`bash run_sgd_diag_tabular.sh smoke|full` を実行するだけでよい (起動時に自動で
`git pull` する。サブディレクトリからの実行でも `.git` はリポジトリルート側を
正しく検出する)。

## 実行前に確認すること

- `run_sgd_diag_tabular.sh` 冒頭の `GPU="?"` を、実際に空いている GPU 番号に
  書き換えること (本実験 Axis A/B/C と競合しない側の GPU)。
- `DATA_ROOT` は Axis B 本実験と同じ `/data01/s_hihara/tabular_data` を指す
  前提にしてある。ラボサーバでの実際の配置と食い違う場合は書き換えること。

## データセットの選定理由

対象は Covertype のみ (3データセット中、既存の AdamW 結果で最も明確に
`interventions=0` かつ `ratio_max<1.0` (常時) という「完全無介入」を示している
ため、SGD に変えたときの変化が一番見えやすい。加えて 20epoch・大きいバッチ
[1024] で最も実行コストが低い)。時間が許せば Adult・California Housing にも
拡張する。

## 進め方 (2段階)

1. `smoke`: seed=0 のみで `ctrl` / `cosine` を実行し、SGD (lr=0.1) で学習が
   発散せず妥当に収束するか、CTRL のログ (kurtosis・current_mult の推移) を
   確認する。
2. `smoke` の結果が問題なければ `full` で seed=0,1,2 の3 seed 分を実行し、
   既存の Axis B (AdamW) 結果 (`project_tabular_experiments.md` 参照、3 seed
   とも `interventions=0`) と比較する。

## 見るべき指標

- 学習が発散していないか (loss が NaN にならない、val 指標が改善しているか)。
- W&B の `ctrl_trace` 内 `current_mult` / `ratio` / `interventions` の推移。
  AdamW 版で 0 だった `interventions` が SGD 版で増えていれば、
  「optimizer 種別 (SGD/AdamW) が CTRL の無介入現象の主因」という仮説を支持する
  材料になる。変わらなければ、アーキテクチャ (Transformer/Attention) 側の性質を
  疑う根拠になる。

## 位置づけ

論文の Limitations/Future Work 節、および CTRL v6 (baseline 設計見直し) の
設計スコープを決めるための診断材料。詳細な経緯は Cowork プロジェクトメモリの
`project_ctrl_v6_design.md` を参照。
