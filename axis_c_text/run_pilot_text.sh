#!/usr/bin/env bash
# ============================================================================
# run_pilot_text.sh — テキストデータ実験(Axis C) 小規模pilot実行スクリプト
#
# ★2026-08-30変更: このスクリプト自身を含む一式(train_text_lstm.py・
#   gala_optimizer.py・cwgd_controller.py)をGitHubの公開リポジトリで管理する
#   方式に切り替えた。従来のOneDrive経由の手動コピーで繰り返し発生していた
#   問題(ファイル破損・謎のヘッダー行混入・md5不一致、gala_optimizer.py/
#   cwgd_controller.pyの配置漏れによるModuleNotFoundError)を解消するため。
#   起動時に自分自身が置かれている場所(=cloneしたリポジトリのルート)を
#   自動検出し、まず`git pull`で最新化してから実行する。したがって以後は
#   コードを更新した場合もラボサーバ側でファイルを個別コピーし直す必要はなく、
#   次回`bash run_pilot_text.sh`を実行するだけで自動的に最新版が反映される。
#
# ★2026-08-30(続き): リポジトリ shoki-hihara/KurtosisController は、将来的に
#   全実験コードを集約する方針のため、axisごとのサブディレクトリ構成に変更された。
#   このファイル(train_text_lstm.py等一式)は `axis_c_text/` サブディレクトリに
#   置かれている(リポジトリ直下ではない)。これに伴い、旧バージョンの
#   「`.git`が自分と同じディレクトリにあるか」で判定していたリポジトリ検出ロジックは
#   (.gitはリポジトリ直下にしかないため)サブディレクトリからは検出できず、
#   git pullが黙って実行されなくなる不具合があったため、下記-1節を修正した。
#
#   【初回セットアップ(このリポジトリをまだcloneしていない場合、最初の1回だけ)】
#     git clone https://github.com/shoki-hihara/KurtosisController.git \
#         /data01/s_hihara/KurtosisController
#     cd /data01/s_hihara/KurtosisController/axis_c_text
#     bash run_pilot_text.sh > pilot_log_text.txt 2>&1
#
#   【他axis用に既にこのリポジトリをcloneしている場合】
#     そのディレクトリで `git pull` した上で axis_c_text/ に入ればよい。
#     改めてcloneし直す必要はない。
#
#   【2回目以降】
#     cd <cloneしたディレクトリ>/axis_c_text
#     bash run_pilot_text.sh > pilot_log_text.txt 2>&1
#     cat pilot_log_text.txt
#
# モデルは Zaremba et al. 2014 (arXiv:1409.2329) の標準的な正則化LSTM
# (medium設定: 2層LSTM、非再帰結合のみdropout)。当初検討していたAWD-LSTM
# (DropConnect+AR/TAR) から、実装リスク低減とCTRL計測との交絡回避のため
# 2026-08-23にこちらへ変更した (train_text_lstm.py の冒頭docstring参照)。
#
# ★GPU割当 (2026-08-29、ユーザー確認済み): GPU1が空いたためGPU1で実行する。
#   ただし2枚とも他の人と共有しているGPUなので、実行の都度、直前の状況を
#   軽く確認してから流すこと (nvidia-smi 等)。
#
# 目的:
#   1. PTB (wojzaremba/lstm由来のローカルファイル)・WikiText-2 (HuggingFace datasets,
#      Salesforce/wikitext) が想定通り読み込めるか確認。
#      ★2026-08-29追記: ptb-text-only/ptb_text_only はスクリプト型データセットであり、
#      datasets>=4.0 ではスクリプト型ローディングが完全廃止されているため恒久的に
#      失敗することが判明した (バージョンフラグでは回避不可)。そのためPTBは
#      --data_source local に切り替え、原著Zaremba et al. 2014実装リポジトリから
#      自動取得する方式に変更した。詳細は下記「1. データソースの準備」を参照。
#   2. LSTMの実GPUでのVRAM使用量を実測 (20GB中どの程度使うか。
#      hidden_size=650・2層と画像・表データよりさらに軽量なはずだが未検証)
#   3. burn-in直後 (PTBは約1.5epoch、WT2は1epoch未満) にkurtosisのspike→
#      over-dampingが起きていないか確認 (他モダリティよりepoch換算でかなり
#      早いタイミングで来るので、pilot序盤のログを特に注意して見ること)
#   4. k_t列とk_t_post_normclip列が一致するか確認 (グローバルノルムクリップは
#      kurtosisを変えないという理論予想を、このプロジェクトのCowork側smoke
#      testでは合成データ上で確認済み。実データでも一致すれば裏付けになる)
# ============================================================================
set -e

# ----------------------------------------------------------------------------
# -1. 自分自身のいる場所を検出し、リポジトリなら最新化する
# ----------------------------------------------------------------------------
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ★2026-08-30修正: このファイルはリポジトリ直下ではなく axis_c_text/ サブディレクトリ
# に置かれるようになったため (.git はリポジトリルートにあり、ここには無い)、
# 従来の `-d "${SELF_DIR}/.git"` ではサブディレクトリから検出できなかった
# (常にelse節に落ちてgit pullが実行されない不具合があった)。
# `git rev-parse --is-inside-work-tree` でgit管理下のどこかにいるかを判定する
# 方式に変更(サブディレクトリからでも正しくリポジトリルートを検出する)。
if git -C "${SELF_DIR}" rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "=== リポジトリを最新化: git -C ${SELF_DIR} pull ==="
    git -C "${SELF_DIR}" pull
else
    echo "!!! ${SELF_DIR} はgitリポジトリではないようです。"
    echo "!!! (初回セットアップがまだの場合は、このファイル冒頭のコメントを参照して"
    echo "!!!  git clone してから実行してください)"
fi

SCRIPT="${SELF_DIR}/train_text_lstm.py"
GPU="1"   # ★2026-08-29確定。GPUの割当状況が変わったらここを書き換えること
DATA_ROOT="/data01/s_hihara/text_data"   # ★データはリポジトリに含めない(gitignore対象)

if [ "${GPU}" = "?" ]; then
    echo "ERROR: GPU番号が未設定です。このスクリプト冒頭の GPU=\"?\" を \"0\" か \"1\" に"
    echo "       書き換えてから再実行してください。"
    exit 1
fi

# ----------------------------------------------------------------------------
# 0. 依存ライブラリの確認
# ----------------------------------------------------------------------------
python3 -c "import datasets" 2>/dev/null || pip install datasets --break-system-packages

mkdir -p ./pilot_results

# ----------------------------------------------------------------------------
# 1. データソースの準備
#
# ★2026-08-29確定 (このthreadで判明): ptb-text-only/ptb_text_only はスクリプト型
#   データセットであり、HuggingFace datasets>=4.0 ではスクリプト型ローディングが
#   完全に廃止されているため、`trust_remote_code=True` を付けても
#   `RuntimeError: Dataset scripts are no longer supported` で失敗する
#   (確認済みcommunity mirror shenlong7/ptb_text_only, FALcon6/ptb_text_only も同じ問題)。
#   これはバージョンフラグでは回避不能な既知の破壊的変更。
#
#   → PTB は --data_source local に切り替え、原著Zaremba et al. 2014 (arXiv:1409.2329)
#     の公式実装リポジトリ (wojzaremba/lstm, GitHub raw) から直接取得する。
#     このファイルの語数(train=887521+42068行, valid=70390+3370行, test=78669+3761行の
#     eos込み合計)は train_text_lstm.py の KNOWN_CORPUS_STATS
#     (train≈929589, valid≈73760, test≈82430) と完全一致することをCowork側で確認済み
#     — 前処理レベルが公式AWD-LSTM実装の getdata.sh 相当データと同一であることの裏付け。
#     GitHub raw content配信のため、原著vutbr.cz FTPミラーより到達性が高いと期待できる。
#
#   WikiText-2 (Salesforce/wikitext, wikitext-2-v1) はParquet形式でスクリプト型では
#   ないため、引き続き --data_source hf を使う (2026-08-29 pilotで動作確認済み)。
# ----------------------------------------------------------------------------
PTB_DIR="${DATA_ROOT}/penn"
if [ ! -f "${PTB_DIR}/train.txt" ] || [ ! -f "${PTB_DIR}/valid.txt" ] || [ ! -f "${PTB_DIR}/test.txt" ]; then
    echo "=== PTBデータを wojzaremba/lstm (原著Zaremba et al. 2014実装) から取得 ==="
    mkdir -p "${PTB_DIR}"
    BASE="https://raw.githubusercontent.com/wojzaremba/lstm/master/data"
    curl -fSL "${BASE}/ptb.train.txt" -o "${PTB_DIR}/train.txt"
    curl -fSL "${BASE}/ptb.valid.txt" -o "${PTB_DIR}/valid.txt"
    curl -fSL "${BASE}/ptb.test.txt"  -o "${PTB_DIR}/test.txt"
    echo "  取得完了: ${PTB_DIR}"
    wc -l "${PTB_DIR}"/{train,valid,test}.txt
else
    echo "=== PTBデータは既に ${PTB_DIR} に存在するためダウンロードをスキップ ==="
fi

echo ""
echo "=== WikiText-2 の HuggingFace datasets (Salesforce/wikitext) 読み込みを確認 ==="
python3 -c "
from datasets import load_dataset
ds2 = load_dataset('Salesforce/wikitext', 'wikitext-2-v1')
print('WikiText-2 OK:', {k: len(v) for k, v in ds2.items()})
" || {
    echo ""
    echo "!!! WikiText-2 の HuggingFace datasets 経由の読み込みにも失敗しました。"
    echo "!!! この場合は --data_source local に切り替え、原著 getdata.sh 相当のデータを"
    echo "!!! ${DATA_ROOT}/wikitext-2/{train,valid,test}.txt に配置してください:"
    echo "!!!   https://github.com/salesforce/awd-lstm-lm/blob/master/getdata.sh"
    echo "!!!   (このスクリプトを実行する代わりに、S3ミラーが生きているか確認すること:"
    echo "!!!    https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)"
    exit 1
}

# ----------------------------------------------------------------------------
# 2. PTB pilot (主データセット。burn-in完了(約1.5epoch)後もしばらく挙動を見たい
#    ので epochs=15に設定。まずCTRLとCOSINEのみで挙動を確認)
# ----------------------------------------------------------------------------
echo ""
echo "############ [1/2] PTB pilot (GPU${GPU}) ############"
python3 "${SCRIPT}" --dataset ptb --protocol noes --schedulers ctrl cosine --seeds 0 \
    --epochs 15 --device cuda --cuda_visible_devices "${GPU}" \
    --data_root "${DATA_ROOT}" --data_source local --save_dir ./pilot_results/ptb

# ----------------------------------------------------------------------------
# 3. WikiText-2 pilot (burn-in完了は1epoch未満。epochs=15で確認)
# ----------------------------------------------------------------------------
echo ""
echo "############ [2/2] WikiText-2 pilot (GPU${GPU}) ############"
python3 "${SCRIPT}" --dataset wikitext2 --protocol noes --schedulers ctrl cosine --seeds 0 \
    --epochs 15 --device cuda --cuda_visible_devices "${GPU}" \
    --data_root "${DATA_ROOT}" --data_source hf --save_dir ./pilot_results/wikitext2

echo ""
echo "=== pilot完了 ==="
echo "確認すること:"
echo "  1. 各runの標準出力にある '[GPU] peak_memory_allocated=' の行 (VRAM実測値)"
echo "  2. ./pilot_results/<dataset>/<dataset>_ctrl_seed0_ctrl_trace.csv の"
echo "     'ratio'列・'current_mult'列を、burn-in完了直後(action列が"
echo "     'burnin'から'continuous_monitor'/'continuous_decay'に変わる行)付近で確認。"
echo "     ratioが異常に跳ね上がりcurrent_multがmin_mult=0.2に張り付いたまま戻らない"
echo "     場合はImageNet-100と同様のover-damping。ハイパラは再チューニングせず"
echo "     現象として記録する。"
echo "  3. 同CSVの'k_t'列と'k_t_post_normclip'列が一致しているか(理論上一致するはず。"
echo "     大きく乖離していたら勾配クリッピングの実装を疑うこと)"
echo "  4. val/perplexityが学習中に発散(inf/nan)していないか"
echo "  5. reach_epoch / best_val_epoch がCTRL burn-in完了(PTB≈1.5epoch, WT2≈0.67epoch)"
echo "     より前に来ていないか (標準出力のWARNING行を確認。特にWT2は1epoch未満で"
echo "     burn-inが終わるため要注意)"
