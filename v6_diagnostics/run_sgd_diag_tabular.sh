#!/usr/bin/env bash
# ============================================================================
# run_sgd_diag_tabular.sh — CTRL v6 診断実験(表データ+SGD) 実行スクリプト
#
# 目的:
#   Axis B (表データ、FT-Transformer+AdamW) で CTRL がほぼ完全に no-op になる
#   現象について、「CTRL は SGD 系の訓練ダイナミクス向けに設計されており、
#   AdamW 系では機能しにくいのではないか」という仮説を確認する。
#   FT-Transformer のアーキテクチャは変えず、optimizer backbone だけを
#   AdamW → SGD (train_tabular_sgd_diag.py の SGD_DIAG_TABULAR_CONFIG,
#   lr=0.1・momentum=0.9・nesterov=True・wd=0) に差し替えて、CTRL の介入挙動
#   (kurtosis推移・intervention回数) が変化するかを見る。
#   Axis B本実験 (train_tabular_ft_transformer.py) とは別物、結果も混同しない。
#
#   このスクリプトは shoki-hihara/KurtosisController リポジトリの一部
#   (https://github.com/shoki-hihara/KurtosisController)。同リポジトリには
#   テキストデータ実験(Axis C, train_text_lstm.py / run_pilot_text.sh)の
#   コードも同居している。自分自身が置かれている場所 (= clone したリポジトリの
#   ルート) を自動検出し、まず `git pull` してから実行する
#   (run_pilot_text.sh と同じ設計)。
#
#   ★このスクリプトはリポジトリの v6_diagnostics/ サブディレクトリに置かれている
#   (リポジトリはAxis A/B/C・CIFAR-100本実験等をaxisごとのサブディレクトリで
#   管理する構成に変更予定、詳細はリポジトリ直下のREADME参照)。
#
#   【初回セットアップ(このリポジトリをまだcloneしていない場合、最初の1回だけ)】
#     git clone https://github.com/shoki-hihara/KurtosisController.git \
#         /data01/s_hihara/KurtosisController
#     cd /data01/s_hihara/KurtosisController/v6_diagnostics
#     bash run_sgd_diag_tabular.sh smoke > sgd_diag_smoke_log.txt 2>&1
#
#   【他axis用に既にこのリポジトリをcloneしている場合】
#     そのディレクトリで `git pull` すれば v6_diagnostics/ 以下が最新化される。
#     改めてcloneし直す必要はない。
#
#   【2回目以降】
#     cd <cloneしたディレクトリ>/v6_diagnostics
#     bash run_sgd_diag_tabular.sh smoke|full > sgd_diag_<mode>_log.txt 2>&1
#
# モード:
#   smoke — seed=0 のみ(ctrl, cosine)。まずこれでSGD(lr=0.1)が発散しないか、
#           CTRLのログが妥当かを確認する。
#   full  — seed=0,1,2 (ctrl, cosine)。smokeで問題なければこちらを実行し、
#           既存のAxis B(AdamW)結果(3 seedともinterventions=0)と比較する。
#
# ★GPU割当: 下記 GPU="?" を、実際に空いているGPU番号に書き換えてから実行する
#   こと(Axis A/B/C本実験と競合しない側のGPUを使う想定、ユーザー確認済み)。
#   2枚とも他の人と共有しているGPUなので、実行の都度 nvidia-smi 等で軽く
#   状況を確認してから流すこと。
# ============================================================================
set -e

# ----------------------------------------------------------------------------
# -1. 自分自身のいる場所を検出し、リポジトリなら最新化する
# ----------------------------------------------------------------------------
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ★このスクリプトはリポジトリ直下ではなく v6_diagnostics/ サブディレクトリに
# 置かれている想定 (.git はリポジトリルートにあり、ここには無い)。
# `-d "${SELF_DIR}/.git"` ではサブディレクトリからは検出できないため、
# `git rev-parse --is-inside-work-tree` でgit管理下のどこかにいるかを判定する。
if git -C "${SELF_DIR}" rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "=== リポジトリを最新化: git -C ${SELF_DIR} pull ==="
    git -C "${SELF_DIR}" pull
fi

MODE="${1:-smoke}"
if [ "${MODE}" != "smoke" ] && [ "${MODE}" != "full" ]; then
    echo "使い方: bash run_sgd_diag_tabular.sh [smoke|full]"
    echo "  smoke (既定) : seed=0 のみ"
    echo "  full         : seed=0,1,2"
    exit 1
fi

DATA_ROOT="/data01/s_hihara/tabular_data"
SCRIPT="${SELF_DIR}/train_tabular_sgd_diag.py"
GPU="?"   # ★実行前に、空いているGPU番号 ("0" か "1") に書き換えること
DATASET="covtype"
SCHEDULERS="ctrl cosine"

if [ "${GPU}" == "?" ]; then
    echo "エラー: GPU=\"?\" のままです。このファイル冒頭の GPU 変数を、"
    echo "        実際に空いているGPU番号に書き換えてから実行してください。"
    exit 1
fi

if [ "${MODE}" == "smoke" ]; then
    SEEDS="0"
else
    SEEDS="0 1 2"
fi

mkdir -p "${SELF_DIR}/sgd_diag_results/${DATASET}_${MODE}"

echo "=== ${DATA_ROOT} 以下で見つかった ${DATASET} の場所 ==="
find "${DATA_ROOT}" -maxdepth 4 -type d -iname "${DATASET}" -o -type d -iname "covertype"

echo ""
echo "############ CTRL v6 診断実験: ${DATASET}, mode=${MODE}, seeds=${SEEDS} (GPU${GPU}) ############"
python3 "${SCRIPT}" \
    --dataset "${DATASET}" --protocol noes \
    --schedulers ${SCHEDULERS} --seeds ${SEEDS} \
    --epochs 20 --device cuda --cuda_visible_devices "${GPU}" \
    --data_root "${DATA_ROOT}" --data_source official \
    --optimizer_override sgd \
    --save_dir "${SELF_DIR}/sgd_diag_results/${DATASET}_${MODE}"

echo ""
echo "=== ${MODE} 完了 ==="
echo "確認すること:"
echo "  1. 上記ログでlossが発散(NaN/Inf)していないか"
echo "  2. ${SELF_DIR}/sgd_diag_results/${DATASET}_${MODE}/${DATASET}_noes_results.csv"
echo "  3. W&B の project 'KurtosisEWMController-tabular-${DATASET}-noes-sgddiag' 内、"
echo "     CTRL runの ctrl_trace (current_mult/ratio/interventions) の推移。"
echo "     既存のAxis B(AdamW)結果は interventions=0 (3 seedとも) だったので、"
echo "     ここで interventions>0 になっているかどうかが仮説の直接の検証材料。"
if [ "${MODE}" == "smoke" ]; then
    echo "  smokeの結果が問題なければ次は: bash run_sgd_diag_tabular.sh full"
fi
