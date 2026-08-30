import os

os.environ["TMPDIR"]             = "/data01/s_hihara/tmp"
os.environ["WANDB_DIR"]          = "/data01/s_hihara/wandb"
os.environ["WANDB_CACHE_DIR"]    = "/data01/s_hihara/wandb/cache"
os.environ.setdefault("HF_DATASETS_CACHE", "/data01/s_hihara/huggingface/datasets")

import wandb
# W&B は事前に `wandb login` で認証しておくこと（.netrc、キーはコードに書かない）

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from gala_optimizer import GALAController
from cwgd_controller import CWGDController

# =====================================================================
# GPU 選択 (train_tabular_ft_transformer.py と同一パターン)
#
# 優先順位: (1) --cuda_visible_devices CLI引数 (__main__ 内で明示的に上書き)
#         > (2) シェルで事前に設定された CUDA_VISIBLE_DEVICES 環境変数
#         > (3) 下記 DEFAULT_CUDA_VISIBLE_DEVICES (このファイルにしか無い場合の
#               フォールバック)
#
# setdefault() を使うことで、外側 (シェル) からの指定を尊重しつつ、未指定時
# のみこのデフォルト値を使う。★2026-08-29、ユーザー確認: GPU1が空いたため
# GPU1で実行する (DEFAULT_CUDA_VISIBLE_DEVICES="1")。それでも実行のたびに
# Axis A/Bの最新の利用状況を確認し、必要なら --cuda_visible_devices で
# 明示的に上書きすること (2枚のGPUは他の人とも共有しているため)。
# =====================================================================

DEFAULT_CUDA_VISIBLE_DEVICES = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_CUDA_VISIBLE_DEVICES)

"""
train_text_lstm.py — CTRL v5 汎用性検証（テキストデータ, Axis C）

============================================================================
目的
============================================================================
CTRL v5 (ContinuousStateKurtosisController) が画像・表データ以外のモダリティ
（テキスト・RNN）でも同じハイパラのまま機能するかを検証する（汎用性の主張の
根拠）。画像データセット実験 (train_v5_multidataset_schedulers.py) ・表データ
実験 (train_tabular_ft_transformer.py) とは別チャットで並行して進めている実験。

対象データセット : Penn Treebank (PTB, 主) / WikiText-2 (WT2, 拡張)
モデル           : 正則化LSTM (Zaremba, Sutskever & Vinyals 2014,
                   "Recurrent Neural Network Regularization", arXiv:1409.2329;
                   medium設定を採用)。2層LSTM + 非再帰結合のみへのdropout。
タスク           : 言語モデリング (word-level)、評価指標は perplexity (低いほど良い)
比較手法         : CTRL / COSINE / WARMUP_COSINE / ONECYCLE / PLATEAU / STEP

============================================================================
モデル選定の経緯 (2026-08-23、ユーザーとの議論で AWD-LSTM から変更)
============================================================================
当初は AWD-LSTM (Merity et al. 2017, arXiv:1708.02182; 3層LSTM + DropConnect +
Variational/Locked Dropout + Embedding Dropout + AR/TAR正則化) を採用し、公式
実装 (salesforce/awd-lstm-lm, 2022年アーカイブ済み) を最新PyTorchに移植して
CPU上のsmoke testまで完走させていた。しかしユーザーから「公式実装がアーカイブ
済みなら別アーキテクチャを検討すべきか」という指摘を受けて再検討し、以下の
理由から Zaremba et al. 2014 の標準的な正則化LSTM (medium設定) に変更した:

  - archived状態自体は実害ではなかった (移植は1件の互換性ハックのみで完了し、
    smoke testも全6手法で成功していた)。しかし、CTRLとの比較枠組みを優先して
    NT-ASGD と可変長BPTTという、AWD-LSTMがtest perplexity 57.3を達成する主要因
    となっていた2つのトリックを既に不採用にしていたため、残る差分 (DropConnect
    + AR/TAR) を維持する意義が薄れていた。
  - DropConnect は再帰結合の重み行列に毎ステップノイズを注入するため、CTRLが
    観測する勾配ノルム分布のkurtosis自体に影響しうる。「RNNアーキテクチャ特有
    の勾配動特性」と「DropConnectノイズの影響」が交絡するリスクがあった。
  - Zaremba et al. 2014 は AWD-LSTM論文自身が比較表で引用する、最も基礎的で
    広く引用されている「標準的な正則化LSTM」のベースラインであり、「文献で
    実際に使われている標準的アーキテクチャを採用する」というプロジェクト方針
    (カスタムベンチマーク禁止) を満たしつつ実装をシンプルにできる。
  - 中心的な工夫は「dropoutを非再帰結合のみに適用する」(再帰結合=hidden-to-
    hidden遷移行列にはdropoutを掛けない) ことであり、これは PyTorch 標準の
    `nn.LSTM(num_layers>1, dropout=p)` がまさにこの意味で層間にdropoutを自動
    適用する仕様と一致するため、DropConnect/Locked Dropout/Embedding Dropout
    のような特殊な自作機構が一切不要になった (flatten_parametersのモンキー
    パッチ等、モダンPyTorchとの互換性ハックも不要)。
  - PTB test perplexityは medium=82.7 / large=78.4 (原論文Table 1) と
    AWD-LSTMの57.3より見劣りするが、本実験の主目的はCTRL vs 他5スケジューラの
    相対比較でありSOTA値自体を追う必要はないため、許容する。

============================================================================
このスレッド (Axis C) で確定した、原論文からの意図的な変更点
============================================================================
以下は「原論文の設定を踏襲する」という一般方針からの逸脱だが、いずれもプロ
ジェクト全体の比較ロジック（CTRL vs 5スケジューラ）と整合させるための意図的
な設計判断であり、恣意的なチューニングではない。論文の再現実験ではなく「CTRL
の汎用性検証」が目的である点を優先した。

1. NT-ASGD に相当する適応的最適化切替は使わず、標準 SGD に統一
   (2026-08-23、ユーザー確認済み。これは元々AWD-LSTM検討時の決定だが、
   Zaremba et al. 2014 も元々はSGD + 手動LR減衰スケジュールなので、標準SGDを
   使うこと自体は原論文とも整合する。手動LR減衰の代わりにCTRL/COSINE/
   ONECYCLE/PLATEAU/STEP/WARMUP_COSINEの6手法を比較する点が本実験の骨子)

2. 勾配クリッピングと kurtosis 計測の関係 (6節5番の未決事項への回答)
   標準的な `torch.nn.utils.clip_grad_norm_` (グローバルL2ノルムに基づく一律
   リスケール) を使う。excess kurtosis はスケール不変 (k(c·x) = k(x) for c>0)
   なので、全パラメータの勾配を一律にリスケールするこの種のクリップは理論上
   kurtosis の値を一切変えない。したがって「クリップ前 vs クリップ後」の二択
   は、グローバルノルムクリップである限り数学的に無意味 (両者は一致する) —
   これは旧AWD-LSTM版のCPU smoke testで実データ(合成データ)上でも
   誤差0.0で一致することを既に確認済みの知見であり、モデルをシンプル化した
   今回もそのまま成り立つ (DropConnectを廃止したことで、以前あった「重み
   ノイズがkurtosisに交絡する」という残存リスクもなくなった)。
   本スクリプトでは実際に pre-clip と post-normclip 両方の kurtosis を
   ctrl_trace CSV に記録する (k_t列とk_t_post_normclip列)。CTRL に渡すのは
   pre-clip 側 (Axis A/B の `collect_grad_magnitudes(clip_max=None)` と同じ
   流儀)。

============================================================================
未検証事項 (★重要)
============================================================================
このセッション (Cowork, GPU/実データ無し) では、CPU上での合成データによる
end-to-end動作確認 (前処理→モデル構築→CTRL適用込みの訓練ループ→勾配クリップ
→reach判定→best checkpoint選定→test-once評価→W&Bログ→CSV集計) を実施し、
構文・ロジックの両面でエラーなく完走することを確認済み。一方で
(1) HuggingFace datasets 経由の PTB/WikiText-2 取得が実際にラボサーバで動くか
(ptb-text-only/ptb_text_only はスクリプト型データセットであり `datasets`
ライブラリのバージョンによっては `trust_remote_code=True` が必要、または
読み込みスクリプト方式が廃止されている可能性がある。動かない場合は
--data_source local で原著 getdata.sh (salesforce/awd-lstm-lm由来。アーキ
テクチャは変更したがデータ取得元・前処理は同じものを流用できる) 相当の
ローカルファイル読み込みにフォールバックすること)、(2) 実GPUでのVRAM使用量・
1epochあたりの実測時間、(3) 実データでの収束挙動・kurtosisの推移 (burn-in
直後のspike有無など) は未検証。ラボサーバでのpilot実行が次のアクション。

============================================================================
モデルアーキテクチャ・学習レシピ (Zaremba et al. 2014, medium設定を踏襲)
============================================================================
PTB / WikiText-2 共通: emsize=650, nhid=650, nlayers=2, dropout=0.5
                       (非再帰結合のみ。nn.LSTMのdropout引数で実現)、
                       batch_size=20, eval_batch_size=10, bptt=35,
                       base_lr=1.0, clip=5.0 (原論文 Table 1, medium設定)
                       weight tyingは原論文(2014年、Press&Wolf 2016以前)に
                       倣い不採用。初期値は uniform(-0.05, 0.05)。
WikiText-2は原論文が扱っていないデータセットだが、恣意的な per-dataset
チューニングに見えないよう、PTBと同一のmedium設定をそのまま適用している。

============================================================================
CTRLハイパラ (全モダリティ共通・再チューニング禁止、6節の確定事項)
============================================================================
alpha=0.05, burnin_steps=2000, alpha_decay=0.1, beta=0.99, min_mult=0.2,
baseline_quantile=0.5, monotone_decrease=True
PTB: 1epoch ≈ (929,589//20)/35 ≈ 1328 step → burn-in(2000step) ≈ 1.5 epoch相当
WikiText-2: 1epoch ≈ (2,088,628//20)/35 ≈ 2984 step
     → burn-in(2000step) ≈ 0.67 epoch相当
(画像・表データよりはるかに短いepoch数でburn-inが終わる。WikiText-2は1epoch
未満で終わる点に特に注意。reach_epoch/best_val_epochがこれより前に来ていないか、
実行時に必ず確認すること)

============================================================================
使い方
============================================================================
# GPU切替は --cuda_visible_devices で。実行前に Axis A (GPU0) / Axis B (GPU1)
# の利用状況をユーザーに確認すること (★このスレッドでは未確認のため実行保留)。
python3 train_text_lstm.py --dataset ptb --protocol noes \
    --schedulers ctrl cosine --seeds 0 --epochs 20 --device cuda \
    --cuda_visible_devices 1 --data_root /data01/s_hihara/text_data \
    --data_source hf --save_dir ./pilot_results/ptb

# 出力はファイルにリダイレクトしてから確認すること (PowerShell経由でのログ
# 流れ対策): bash run_pilot_text.sh > pilot_log.txt 2>&1 ; cat pilot_log.txt
"""

from typing import List, Optional, Tuple


# =====================================================================
# Kurtosis / CTRL controller
# (train_v5_multidataset_schedulers.py / train_tabular_ft_transformer.py と
#  完全同一。データセット・モダリティに依存しないコアロジックなのでそのまま
#  流用する。★このクラスは一切変更しない。)
# =====================================================================

@torch.no_grad()
def excess_kurtosis(x: torch.Tensor, eps: float = 1e-12) -> float:
    # ★2026-08-30修正 (Axis Cのpilotで発覚): 元の実装は分母を `v ** 2 + eps` として
    # いたが、これはスケールが不整合なバグだった。直前のガード `if v < eps` は v
    # 自体(線形スケール)をepsと比較しているのに対し、分母に足すepsはv**2(2乗
    # スケール)に対して加算されるため、"v はガードを通過する(v > eps) が
    # v**2 は eps を下回る" という範囲 (eps < v < sqrt(eps) ≒ 1e-6) が抜け穴になり、
    # 本来無視できるはずのepsが分母を支配して結果を系統的に -3 側へ歪めていた。
    # LSTM+大語彙embeddingのように勾配絶対値の分散が小さくなりやすい構成
    # (embeddingの疎な勾配で大半の要素が厳密に0になるため)でこれが顕在化し、
    # 理論的にあり得ない値(excess kurtosisは任意の実数値分布で下限-2、
    # Kurt >= Skew^2+1 という古典的不等式による)が大量に発生していた
    # (実データで検証済み: PTB/WikiText-2ともpilotのk_t系列の95%超が-2を下回り、
    # 最小値は-3近傍。float32/float64どちらで計算しても同じ値になることを確認して
    # おり、精度の問題ではなくこのepsのスケール不一致が原因と特定した)。
    # 「v**2+eps」の項からepsを単純に外し、真にゼロ除算になり得るケース
    # (v がほぼ0)は直前の `if v < eps: return 0.0` ガードに任せる形にした。
    x = x.float()
    if x.numel() < 10:
        return float("nan")
    mu = x.mean()
    v = x.var(unbiased=False)
    if v < eps:
        return 0.0
    m4 = ((x - mu) ** 4).mean()
    k = m4 / (v ** 2) - 3.0
    return float(k.item())


@torch.no_grad()
def collect_grad_magnitudes(model: torch.nn.Module, clip_max: float = None) -> torch.Tensor:
    mags = []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        mags.append(g.abs().reshape(-1))
    if not mags:
        return torch.empty(0)
    x = torch.cat(mags)
    if clip_max is not None:
        x = torch.clamp(x, max=clip_max)
    return x


class ContinuousStateKurtosisController:
    """
    v5: kurtosis 状態に連続的に応答する controller。

    threshold / trigger / cooldown / phase なし。calibration (burn-in) と
    continuous adjustment の2局面のみ。

    ★ このクラスはデータセット・モダリティ非依存。汎用性検証のため一切変更しない。
    """

    def __init__(
        self,
        alpha: float = 0.05,
        burnin_steps: int = 500,
        alpha_decay: float = 0.5,
        beta: float = 0.99,
        min_mult: float = 0.1,
        baseline_quantile: float = 0.5,
        monotone_decrease: bool = True,
    ):
        self.alpha = alpha
        self.burnin_steps = burnin_steps
        self.alpha_decay = alpha_decay
        self.beta = beta
        self.min_mult = min_mult
        self.baseline_quantile = baseline_quantile
        self.monotone_decrease = monotone_decrease

        self.step = 0
        self.k_ewm = None

        self._burnin_kewm = []
        self.baseline = None

        self.current_mult = 1.0
        self.original_base_lrs = None
        self.base_lr = None

        self.interventions = 0
        self._mult_drops_log = []

        self._since_last_log_mult_change = 0

    @torch.no_grad()
    def update(self, k_t, optimizer, scheduler=None) -> dict:
        self.step += 1

        if self.original_base_lrs is None and scheduler is not None:
            self.original_base_lrs = list(scheduler.base_lrs)
            self.base_lr = self.original_base_lrs[0]
        if self.base_lr is None and optimizer is not None:
            self.base_lr = optimizer.param_groups[0]["lr"]

        if not (k_t == k_t):
            return self._make_info(
                k_t=k_t, ratio=float("nan"),
                target_mult=float("nan"), action="nan_kurtosis",
            )

        if self.k_ewm is None:
            self.k_ewm = float(k_t)
        else:
            self.k_ewm = self.alpha * float(k_t) + (1.0 - self.alpha) * self.k_ewm

        if self.step <= self.burnin_steps:
            self._burnin_kewm.append(float(self.k_ewm))
            if self.step == self.burnin_steps:
                self._finalize_baseline()
            return self._make_info(
                k_t=k_t, ratio=float("nan"),
                target_mult=1.0, action="burnin",
            )

        if self.baseline is None or self.baseline <= 0:
            return self._make_info(
                k_t=k_t, ratio=float("nan"),
                target_mult=1.0, action="no_baseline",
            )

        ratio = self.k_ewm / self.baseline
        excess = max(0.0, ratio - 1.0)

        target_mult = max(self.min_mult, math.exp(-self.alpha_decay * excess))

        smoothed_mult = self.beta * self.current_mult + (1.0 - self.beta) * target_mult

        prev_mult = self.current_mult
        if self.monotone_decrease:
            self.current_mult = min(self.current_mult, smoothed_mult)
        else:
            self.current_mult = smoothed_mult

        action = "continuous_monitor"
        if (prev_mult - self.current_mult) > 1e-4:
            action = "continuous_decay"
            self.interventions += 1
            self._since_last_log_mult_change = 1

        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [
                ob * self.current_mult for ob in self.original_base_lrs
            ]
        if optimizer is not None:
            for pg in optimizer.param_groups:
                if prev_mult > 0:
                    pg["lr"] = pg["lr"] * (self.current_mult / prev_mult)

        info = self._make_info(
            k_t=k_t, ratio=ratio,
            target_mult=target_mult, action=action,
        )
        info["current_lr"] = optimizer.param_groups[0]["lr"] if optimizer else None
        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            info["base_lrs"] = list(scheduler.base_lrs)
        return info

    def _finalize_baseline(self):
        if len(self._burnin_kewm) < 30:
            self.baseline = None
            return
        vals = torch.tensor(self._burnin_kewm, dtype=torch.float32)
        self.baseline = float(torch.quantile(vals, self.baseline_quantile).item())

    def _make_info(self, k_t, ratio, target_mult, action):
        return {
            "step": self.step,
            "k_t": float(k_t) if k_t == k_t else float("nan"),
            "k_ewm": self.k_ewm,
            "baseline": self.baseline,
            "ratio": ratio if isinstance(ratio, float) else float(ratio),
            "target_mult": target_mult if isinstance(target_mult, float) else float(target_mult),
            "current_mult": self.current_mult,
            "interventions": self.interventions,
            "action": action,
        }

    def consume_log_flags(self):
        out = {"since_last_log_mult_change": self._since_last_log_mult_change}
        self._since_last_log_mult_change = 0
        return out


# CTRL ハイパラ: 全データセット・全モダリティ共通・再チューニングなし (★確定値、画像/表データ実験と完全同一)
CTRL_CONFIG = dict(
    alpha=0.05,
    burnin_steps=2000,
    alpha_decay=0.1,
    beta=0.99,
    min_mult=0.2,
    baseline_quantile=0.5,
    monotone_decrease=True,
)


# =====================================================================
# GALA / CWGD — LSTM (Zaremba et al. 2014, truncated BPTT) 対応
#
# 画像・表データ軸と異なり、本軸のモデル forward は `model(data, hidden)`
# という2引数シグネチャで、かつ hidden state (truncated BPTT の
# エントリー時点のもの) を「同一ミニバッチ」の一部として GALA/CWGD の
# 追加 forward/backward に再利用する必要がある。以下、両コントローラの
# 既存 API (forward_fn / per_sample_loss_fn フック、tabular軸のFT-Transformer
# 統合で確立したのと同じアダプタパターン) を使い、コントローラ本体
# (gala_optimizer.py / cwgd_controller.py) は一切変更しない。
#
# ---------------------------------------------------------------------
# GALA: forward_fn(model, data, hidden) -> output.view(-1, ntoken)
# ---------------------------------------------------------------------
# 呼び出し側 (train_one_seed_text) は、repackage_hidden 直後の「エントリー
# 時点の hidden state」を hidden_in として別変数に保持し、
# `gala.step(model, (data, hidden_in), targets, criterion, optimizer,
#            forward_fn=forward_lstm_for_gala)` のように渡す。これにより
# GALA 内部の追加逆伝播 (旧パラメータ theta_{n-1} + 同一バッチ) が、
# 主 forward と同じ (data, hidden_in) ペアを再利用する
# (引継ぎタスクの設計方針 (a) そのもの)。
#
# また Zaremba et al. 2014 のレシピは global-norm 勾配クリッピング
# (clip=5.0) が前提であり、外側ループは g_n をクリップしてから
# gala.step() を呼ぶ。GALA 内部で計算する g_prev がクリップされないと
# L_t/alignment の計算で g_n (クリップ済み) と g_prev (未クリップ) の
# スケールが不整合になるため、GALAController に追加した grad_clip 引数
# (gala_optimizer.py 参照) で g_prev にも同じ max_norm を適用する。
#
# ---------------------------------------------------------------------
# CWGD: batch-first 転置アダプタ
# ---------------------------------------------------------------------
# CWGDController の _compute_cwgd_t は「inputs タプルの各要素は dim=0 が
# バッチ次元」という前提で per-sample vmap を行うが、LSTM の data は
# (seq_len, batch)、hidden は (nlayers, batch, nhid) で dim=1 がバッチ
# 次元になっている。そこで呼び出し側で data/h0/c0/target を
# batch-first (dim0=batch) に転置してから update() に渡す:
#   data_bf   = data.transpose(0, 1)                # (batch, seq_len)
#   h0_bf/c0_bf = h0/c0.transpose(0, 1)              # (batch, nlayers, nhid)
#   target_bf = targets.view(seq_len, batch).transpose(0, 1)  # (batch, seq_len)
# (targets は get_batch() が (seq_len*batch,) に flatten 済みなので、
#  一旦 (seq_len, batch) に戻してから転置する)
#
# forward_fn (Hutchinson refresh 用) は batch-first 入力を seq-first に
# 戻して通常の forward を行い、出力を (batch, ntoken, seq_len) に
# permute する。nn.CrossEntropyLoss の K 次元入力形式 (N, C, d1, ...) に
# 一致させることで、target を (batch, seq_len) のまま渡せる。
# これは標準の (seq_len*batch,) flatten 経由の loss と完全に同一の値に
# なることを検証済み (test_lstm_style.py)。
#
# per_sample_loss_fn (per-sample 勾配 vmap 用) は「1バッチ内の1系列」を
# 1サンプルとみなし、その系列内の平均トークン NLL を返す。原論文は
# i.i.d. サンプルの回帰/分類のみを想定しており系列モデルへの拡張には
# 言及がないため、これは本実装独自の設計判断であることを明記する。
#
# ★ 実装上の注意 (torch.func.vmap の既知の制約): PyTorch の LSTM は
# vmap のバッチングルールが未実装 (aten::mkldnn_rnn_layer 等) のため、
# vmap は遅いフォールバック経路 (内部でループ的に処理) を使う。これは
# CIFAR-100/ResNet18 等で計測した「CWGD は plain SGD の 12〜17倍」より
# さらに悪化する可能性が高く、テキスト軸での CWGD パイロットは
# 必ず最小構成 (小 epoch・小 batch) で計算コストを確認してから
# 本実験に進むこと (引継ぎタスクの pilot-first 指示がテキスト軸では
# 一層重要になる)。
# =====================================================================

GALA_TEXT_CONFIG_KEYS = ("momentum", "nesterov", "weight_decay")  # 参照用 (下のrun_schedulers内で動的構築)

CWGD_CONFIG = dict(
    alpha=1.0,
    num_probes=20,
    hutchinson_delta=1e-3,
    refresh_interval_steps=None,  # None -> train_one_seed_text 内で Delta≈T/8 epoch 相当を自動導出
    lambda_floor=1e-6,
    subsample_size=None,
)


def forward_lstm_for_gala(model, data, hidden):
    """GALA用 forward_fn。criterion は output.view(-1, ntoken) を期待する
    (train_one_seed_text の主 forward と同じ reshape 規約)。"""
    output, _ = model(data, hidden)
    return output.view(-1, output.size(-1))


def forward_lstm_for_cwgd_refresh(model, data_bf, h0_bf, c0_bf):
    """CWGDのHutchinson refresh用 forward_fn。batch-first入力をseq-firstに
    戻してforwardし、出力を (batch, ntoken, seq_len) に並べ替える
    (nn.CrossEntropyLossのK次元形式に合わせ、targetを (batch, seq_len) の
    まま渡せるようにするため。標準のflatten経由lossと数値的に完全一致する
    ことをtest_lstm_style.pyで検証済み)。"""
    data = data_bf.transpose(0, 1)
    h0 = h0_bf.transpose(0, 1)
    c0 = c0_bf.transpose(0, 1)
    output, _ = model(data, (h0, c0))          # (seq_len, batch, ntoken)
    return output.permute(1, 2, 0)              # (batch, ntoken, seq_len)


def lstm_per_sample_loss_fn(model, params, buffers, data_1, h0_1, c0_1, target_1):
    """CWGDのper-sample勾配(vmap)用。1バッチ内の1系列を1サンプルとみなし、
    その系列内の平均トークンNLLを返す (本LSTM軸独自の設計判断。上記コメント参照)。"""
    data_in = data_1.unsqueeze(1)      # (seq_len, 1)
    h0_in = h0_1.unsqueeze(1)          # (nlayers, 1, nhid)
    c0_in = c0_1.unsqueeze(1)
    out, _ = functional_call(model, (params, buffers), (data_in, (h0_in, c0_in)))
    out_flat = out.reshape(-1, out.size(-1))   # (seq_len, ntoken)
    return nn.functional.cross_entropy(out_flat, target_1)


# =====================================================================
# Seed
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# LSTM モデル本体 (Zaremba, Sutskever & Vinyals 2014, arXiv:1409.2329,
# "Recurrent Neural Network Regularization" の medium 設定)
#
# 中心的な工夫は「dropoutを非再帰結合のみに適用する」こと (再帰結合=
# hidden-to-hidden遷移行列にはdropoutを掛けない)。PyTorch標準の
# nn.LSTM(num_layers>1, dropout=p) はまさにこの意味で層間にdropoutを自動適用
# する仕様なので、DropConnect/Locked Dropout/Embedding Dropoutのような特殊な
# 自作機構は不要 — 素の nn.LSTM + nn.Dropout(embedding直後・出力直前) のみで
# 原論文のレシピを再現できる。weight tyingは原論文(2014年、Press&Wolf 2016
# 以前)に倣い不採用。
# =====================================================================

class RNNModel(nn.Module):
    """Zaremba et al. 2014 medium設定の正則化LSTM (2層、非再帰結合のみdropout)。"""

    def __init__(self, ntoken: int, ninp: int, nhid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # nn.LSTMのdropout引数: 最終層を除く各層の"出力"にdropoutを適用する
        # (=層間の非再帰結合のみにかかり、再帰結合(hidden-to-hidden)には
        # かからない)。これが原論文の中心的な工夫そのもの。
        self.rnn = nn.LSTM(ninp, nhid, num_layers=nlayers,
                            dropout=(dropout if nlayers > 1 else 0.0), batch_first=False)
        self.decoder = nn.Linear(nhid, ntoken)

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

        self.init_weights()

    def init_weights(self):
        # Zaremba et al. 2014: 全パラメータを uniform(-0.05, 0.05) で初期化
        initrange = 0.05
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)

    def init_hidden(self, bsz: int, device):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid).to(device),
            weight.new_zeros(self.nlayers, bsz, self.nhid).to(device),
        )

    def forward(self, input: torch.Tensor, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0) * output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, hidden


def repackage_hidden(h):
    """truncated BPTT: 前バッチのhidden stateを計算グラフから切り離す。"""
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(repackage_hidden(v) for v in h)


# =====================================================================
# データ: Dictionary / Corpus (word-level, 原著 data.py と同じ前処理:
# 行ごとに単語分割 + 行末に <eos> を追加)
# =====================================================================

class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    """train_lines/valid_lines/test_linesは「1要素=1行(文)」のstrリスト。
    語彙はtrainのみから構築する (原著と同じ; PTB/WikiText-2の標準前処理済み
    コーパスは既に<unk>置換済みなので、val/testのOOVは通常発生しない想定)。"""

    def __init__(self, train_lines: List[str], valid_lines: List[str], test_lines: List[str]):
        self.dictionary = Dictionary()
        # <unk> は原著コーパスに既に含まれているはずだが、保険として先に確保しておく
        self.dictionary.add_word("<unk>")
        self.train = self._tokenize(train_lines, build_vocab=True)
        self.valid = self._tokenize(valid_lines, build_vocab=False)
        self.test = self._tokenize(test_lines, build_vocab=False)

    def _tokenize(self, lines: List[str], build_vocab: bool) -> torch.Tensor:
        ids_list = []
        unk_idx = self.dictionary.word2idx["<unk>"]
        oov_count = 0
        for line in lines:
            words = line.split() + ["<eos>"]
            for w in words:
                if build_vocab:
                    idx = self.dictionary.add_word(w)
                else:
                    idx = self.dictionary.word2idx.get(w, None)
                    if idx is None:
                        idx = unk_idx
                        oov_count += 1
                ids_list.append(idx)
        if not build_vocab and oov_count > 0:
            print(f"  [Corpus] warning: {oov_count} OOV tokens mapped to <unk> "
                  f"(train語彙に含まれない単語が val/test に出現。標準コーパスなら"
                  f"通常0のはず。前処理・データソースを確認すること)")
        return torch.tensor(ids_list, dtype=torch.long)


# train/valid/test の既知の目安 (文献値、word-level・<eos>付与後の概算)。
# 実測値との乖離が大きい場合はトークナイズ方式のずれを疑うための参考値
# (Axis Bの OFFICIAL_SPLIT_SIZES と同じ考え方。厳密一致は要求しないソフトチェック)。
KNOWN_CORPUS_STATS = {
    "ptb": dict(vocab_size=10_000, train_tokens=929_589, valid_tokens=73_760, test_tokens=82_430),
    "wikitext2": dict(vocab_size=33_278, train_tokens=2_088_628, valid_tokens=217_646, test_tokens=245_569),
}


def _check_corpus_stats(dataset: str, corpus: "Corpus"):
    expected = KNOWN_CORPUS_STATS.get(dataset)
    if expected is None:
        return
    actual = dict(
        vocab_size=len(corpus.dictionary),
        train_tokens=len(corpus.train), valid_tokens=len(corpus.valid), test_tokens=len(corpus.test),
    )
    print(f"  [Corpus stats] {dataset}: actual={actual} / 文献目安={expected}")
    for k in expected:
        if expected[k] > 0 and abs(actual[k] - expected[k]) / expected[k] > 0.05:
            print(f"  [Corpus stats][WARNING] {k} が文献目安から5%以上乖離 "
                  f"(actual={actual[k]}, expected≈{expected[k]})。トークナイズ方式や"
                  f"データソースのずれの可能性があるので確認すること。")


# =====================================================================
# データ取得: HuggingFace datasets (既定) / ローカルファイル (フォールバック)
# =====================================================================

# 原著 getdata.sh 相当のローカル展開を想定したディレクトリ名候補
# (train.txt/valid.txt/test.txt を直下に持つディレクトリを再帰探索する。
#  Axis Bの教訓: ディレクトリ構造の想定は外れる前提で頑健に作ること)
LOCAL_DATASET_DIRNAME_CANDIDATES = {
    "ptb": ["penn", "ptb", "penn_treebank", "PTB"],
    "wikitext2": ["wikitext-2", "wikitext2", "wt2", "wikitext_2"],
}


def _resolve_local_text_dir(dataset: str, data_root: str) -> Optional[str]:
    candidates = [c.lower() for c in LOCAL_DATASET_DIRNAME_CANDIDATES[dataset]]
    matches = []
    for root, dirs, files in os.walk(data_root):
        base = os.path.basename(root).lower()
        fset = set(f.lower() for f in files)
        if base in candidates and {"train.txt", "valid.txt", "test.txt"}.issubset(fset):
            matches.append(root)
    if matches:
        return matches[0]
    return None


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _load_text_splits_local(dataset: str, data_root: str) -> Tuple[List[str], List[str], List[str]]:
    d = _resolve_local_text_dir(dataset, data_root)
    if d is None:
        tree_lines = []
        for root, dirs, files in os.walk(data_root):
            depth = root[len(data_root):].count(os.sep)
            if depth > 3:
                dirs[:] = []
                continue
            tree_lines.append(f"{'  ' * depth}{os.path.basename(root) or root}/")
        raise FileNotFoundError(
            f"'{dataset}' の train.txt/valid.txt/test.txt が {data_root} 以下に見つかりません。\n"
            f"原著 getdata.sh (https://github.com/salesforce/awd-lstm-lm/blob/master/getdata.sh) "
            f"相当のデータを配置するか、--data_source hf を使ってください。\n"
            f"現在の {data_root} 以下の構成:\n" + "\n".join(tree_lines[:200])
        )
    print(f"  [data] '{dataset}' をローカルファイルから読み込み: {d}")
    return (_read_lines(os.path.join(d, "train.txt")),
            _read_lines(os.path.join(d, "valid.txt")),
            _read_lines(os.path.join(d, "test.txt")))


def _load_text_splits_hf(dataset: str) -> Tuple[List[str], List[str], List[str]]:
    from datasets import load_dataset

    if dataset == "ptb":
        # ★2026-08-29確定: ptb-text-only/ptb_text_only (および確認済みの community mirror
        # shenlong7/ptb_text_only, FALcon6/ptb_text_only 等) はすべてスクリプト型データセット
        # であり、datasets>=4.0 ではスクリプト型ローディングそのものが完全に廃止されている
        # (trust_remote_code=True を付けてもダメ。「not supported anymore」で弾かれる)。
        # そのため PTB は HF 経由では読み込めない既知の問題であり、--data_source local を
        # 使うこと (run_pilot_text.sh が原著Zaremba et al. 2014の公式実装リポジトリ
        # (wojzaremba/lstm, GitHub raw) から自動取得する)。
        try:
            ds = load_dataset("ptb-text-only/ptb_text_only", trust_remote_code=True)
        except TypeError:
            # 古い datasets バージョンには trust_remote_code 引数自体が無い場合がある
            ds = load_dataset("ptb-text-only/ptb_text_only")
        except RuntimeError as e:
            raise RuntimeError(
                "PTB を HuggingFace datasets 経由で読み込めませんでした。"
                "datasets>=4.0 はスクリプト型データセット (ptb-text-only/ptb_text_only を含む) "
                "の読み込みを完全に廃止しています。--data_source local を使ってください "
                "(run_pilot_text.sh 参照)。"
            ) from e
        train_lines = list(ds["train"]["sentence"])
        valid_lines = list(ds["validation"]["sentence"])
        test_lines = list(ds["test"]["sentence"])
    elif dataset == "wikitext2":
        # wikitext-2-v1 (非raw版): <unk>置換済みのword-level版。原著getdata.shが
        # 落としてくる wiki.*.tokens と同じ前処理レベルに対応する。
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
        train_lines = list(ds["train"]["text"])
        valid_lines = list(ds["validation"]["text"])
        test_lines = list(ds["test"]["text"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_lines, valid_lines, test_lines


def load_corpus(dataset: str, data_root: str, data_source: str = "hf") -> "Corpus":
    if data_source == "hf":
        train_lines, valid_lines, test_lines = _load_text_splits_hf(dataset)
    elif data_source == "local":
        train_lines, valid_lines, test_lines = _load_text_splits_local(dataset, data_root)
    else:
        raise ValueError(f"Unknown data_source: {data_source} (choices: hf, local)")
    corpus = Corpus(train_lines, valid_lines, test_lines)
    _check_corpus_stats(dataset, corpus)
    return corpus


def batchify(data: torch.Tensor, bsz: int, device) -> torch.Tensor:
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source: torch.Tensor, i: int, bptt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def steps_per_epoch_from_batchified_len(n_batchified_tokens: int, bptt: int) -> int:
    """batchify後の系列長(train_data.size(0)相当)から、1epoch分の
    `for i in range(0, n-1, bptt)` の実反復回数(=optimizer.step()回数)を計算する。

    ★2026-08-23 smoke testで発見: 単純な `(n - 1) // bptt` (floor除算) は
    range(0, n-1, bptt) の実際の反復回数 (末尾の端数チャンクも1回とカウントする
    ceil除算) と一致せず、OneCycleLR の total_steps 見積もりがずれて
    'Tried to step N times. The specified number of total steps is N-1' という
    実行時エラーになった。total_steps系の値は必ずこの関数経由で計算し、学習
    ループの実際の反復回数と完全に一致させること (教訓1の徹底: 動的パラメータの
    不一致を作らない)。
    """
    n = max(0, n_batchified_tokens - 1)
    if n == 0:
        return 0
    return -(-n // bptt)  # ceil division


# =====================================================================
# スケジューラ (train_v5_multidataset_schedulers.py / train_tabular_ft_transformer.py
# と同一。optimizerの種類に依存しない汎用ロジックなので変更不要)
# =====================================================================

def make_scheduler(scheduler_name: str, optimizer, **kwargs):
    name = scheduler_name.lower()

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get("T_max", 200))

    elif name == "onecycle":
        if "total_steps" not in kwargs:
            raise ValueError("OneCycleLR requires 'total_steps' in kwargs")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 1e-3),
            total_steps=kwargs["total_steps"],
            pct_start=kwargs.get("pct_start", 0.1),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4),
            three_phase=kwargs.get("three_phase", False),
        )

    elif name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
            threshold=kwargs.get("threshold", 1e-4),
            min_lr=kwargs.get("min_lr", 1e-7),
        )

    elif name in ("step", "multistep", "steplr", "multisteplr"):
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get("milestones", [60, 90]),
            gamma=kwargs.get("gamma", 0.1),
        )

    elif name in ("warmup_cosine", "warmupcosine"):
        warmup_epochs = kwargs.get("warmup_epochs", 5)
        T_max = kwargs.get("T_max", 100)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, T_max - warmup_epochs),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs],
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def build_effective_scheduler_kwargs(scheduler_name: str, num_epochs: int, total_steps: int) -> dict:
    """★教訓1の一般化対応: T_max/milestones/total_stepsを実行時のnum_epochsから
    毎回動的に導出する。Axis Bで「CTRL側は実行時num_epochsを使うのに、COSINE側
    は初期の暫定値を使い続けてしまう」というバグで3データセット分のpilotが無効
    になった教訓を踏まえ、事前定義のプリセット値ではなく常にここで計算する。"""
    name = scheduler_name.lower()
    if name == "cosine":
        return dict(T_max=num_epochs)
    elif name == "onecycle":
        return dict(max_lr=1.0, pct_start=0.1, anneal_strategy="cos",
                     div_factor=25.0, final_div_factor=1e4, total_steps=total_steps)
    elif name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
        return dict(mode="min", factor=0.5, patience=max(2, int(round(num_epochs * 0.08))),
                     threshold=1e-3, min_lr=1e-3)
    elif name in ("step", "multistep", "steplr", "multisteplr"):
        return dict(milestones=[int(round(num_epochs * 0.6)), int(round(num_epochs * 0.85))], gamma=0.1)
    elif name in ("warmup_cosine", "warmupcosine"):
        return dict(warmup_epochs=max(1, int(round(num_epochs * 0.03))), T_max=num_epochs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# =====================================================================
# データセット別プリセット (★2026-08-23作成、暫定値。要pilot後確定)
#
# CTRL_CONFIGとは異なりこれらは「原論文の設定を踏襲する」方針の対象
# (アーキテクチャ・正則化係数はZaremba et al. 2014 Table 1のmedium設定を
# そのまま採用、再チューニングではない)。num_epochs / reach_threshold /
# ES2のmin_deltaは、原論文が手動LR減衰スケジュールでの停止基準を使っており
# 本実験のCTRL/6スケジューラ比較に直接転用できないため暫定値。
# ★重要 (6節6番の未決事項): min_delta=0.001 はaccuracy/RMSEスケール向けの値
# であり、perplexity(数十〜百のスケール)には小さすぎて意味を持たない。ここでは
# perplexityスケールに合わせた暫定値を置いているが、必ずpilot結果を見て確定
# させること。
# =====================================================================

TEXT_DATASET_CONFIGS = {
    "ptb": dict(
        emsize=650, nhid=650, nlayers=2, dropout=0.5, wdecay=0.0,
        batch_size=20, eval_batch_size=10, bptt=35,
        base_lr=1.0, clip=5.0,
        num_epochs=55,           # ★暫定。要pilot後確定 (burn-inは約1.5epochで完了する見込み)
        reach_threshold=95.0,    # ★暫定 (perplexity, 低いほど良い。原論文medium設定はtest≈82.7 [手動LR減衰込み]。
                                  #   本実験は6スケジューラ比較のため若干緩めに設定。要pilot後確定)
        es2_min_delta=0.5,       # ★暫定 (perplexity 0.5ポイント。accuracy/RMSEスケール向けの0.001は不適切。要pilot後確定)
    ),
    "wikitext2": dict(
        # 原論文はWikiText-2を扱っていないため、恣意的なper-datasetチューニング
        # に見えないよう、PTBと同一のmedium設定をそのまま適用する。
        emsize=650, nhid=650, nlayers=2, dropout=0.5, wdecay=0.0,
        batch_size=20, eval_batch_size=10, bptt=35,
        base_lr=1.0, clip=5.0,
        num_epochs=55,           # ★暫定。要pilot後確定 (burn-inは1epoch未満で完了する見込み)
        reach_threshold=110.0,   # ★暫定 (WikiText-2は語彙数がPTBの3倍超(約33k vs 10k)でperplexityが
                                  #   一般に高くなる傾向があるため、PTBよりやや緩めに設定。要pilot後確定)
        es2_min_delta=0.5,       # ★暫定。要pilot後確定
    ),
}


# =====================================================================
# 評価 (perplexity)
# =====================================================================

@torch.no_grad()
def evaluate_perplexity(model: RNNModel, data_source: torch.Tensor, eval_batch_size: int,
                         bptt: int, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hidden = model.init_hidden(eval_batch_size, device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt)
        output, hidden = model(data, hidden)
        hidden = repackage_hidden(hidden)
        loss = criterion(output.view(-1, output.size(-1)), targets)
        total_loss += loss.item()
        total_tokens += targets.numel()
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return ppl


# =====================================================================
# 訓練 loop 1 run 分
# =====================================================================

def train_one_seed_text(
    *,
    dataset: str,
    corpus: "Corpus",
    device,
    seed: int,
    run_name: str,
    project: str,
    scheduler_name: str,
    scheduler_kwargs: dict,
    num_epochs: int,
    reach_threshold: float,
    patience: int,
    min_delta: float,
    log_every: int = 200,
    extra_config: Optional[dict] = None,
    controller_config: Optional[dict] = None,
    gala_config: Optional[dict] = None,
    cwgd_config: Optional[dict] = None,
    save_dir: Optional[str] = None,
    method_label: Optional[str] = None,
) -> dict:
    """1 seed 分の訓練。train_tabular_ft_transformer.py の train_one_seed_tabular
    と骨格 (CTRL適用箇所・reach判定・test-once評価・W&Bロギング) を揃えてある。
    perplexityは低いほど良い指標 (regressionのRMSEと同じ枠組み)。
    controller_config / gala_config / cwgd_config は互いに排他。"""
    set_seed(seed)
    cfg = TEXT_DATASET_CONFIGS[dataset]

    use_gala = gala_config is not None
    use_cwgd = cwgd_config is not None
    use_ctrl = controller_config is not None
    assert sum([use_gala, use_cwgd, use_ctrl]) <= 1, (
        "controller_config / gala_config / cwgd_config は互いに排他です"
    )

    ntokens = len(corpus.dictionary)
    train_data = batchify(corpus.train, cfg["batch_size"], device)
    val_data = batchify(corpus.valid, cfg["eval_batch_size"], device)
    test_data = batchify(corpus.test, cfg["eval_batch_size"], device)

    model = RNNModel(
        ntoken=ntokens, ninp=cfg["emsize"], nhid=cfg["nhid"], nlayers=cfg["nlayers"],
        dropout=cfg["dropout"],
    ).to(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    criterion = nn.CrossEntropyLoss()

    gala_controller = None
    if use_gala:
        # GALA は torch scheduler を使わず、GALAController が毎 step 自律的に
        # optimizer の lr を書き換える (他3軸と同じ設計)。この text 軸は
        # 原論文 Zaremba et al. 2014 のレシピ (momentum=0, wdecay=cfg["wdecay"])
        # を CTRL/他スケジューラと共有しているため、GALA もそれに相乗りさせる
        # (「GALA は momentum に理論的制約がないので比較対象の optimizer 設定を
        #  共有してよい」という引継ぎ仕様書の指示を、この軸では「この軸自身の
        #  CTRL/baseline 設定 = momentum=0」に適用した結果)。
        gala_controller = GALAController(**gala_config)
        optimizer = gala_controller.build_optimizer(model)
        scheduler = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["base_lr"],
                                     momentum=0.0, weight_decay=cfg["wdecay"])
        scheduler = make_scheduler(scheduler_name, optimizer, **scheduler_kwargs)

    controller = ContinuousStateKurtosisController(**controller_config) if use_ctrl else None

    bptt = cfg["bptt"]
    steps_per_epoch = max(1, steps_per_epoch_from_batchified_len(train_data.size(0), bptt))

    cwgd_controller = None
    if use_cwgd:
        cwgd_cfg = dict(cwgd_config)
        if cwgd_cfg.get("refresh_interval_steps") is None:
            refresh_epochs = max(1, num_epochs // 8)
            cwgd_cfg["refresh_interval_steps"] = steps_per_epoch * refresh_epochs
        cwgd_controller = CWGDController(**cwgd_cfg)

    name = scheduler_name.lower()
    is_onecycle = name == "onecycle"
    is_plateau = name in ("plateau", "reduce_on_plateau", "reducelronplateau")

    burnin_epoch_equiv = (controller_config["burnin_steps"] / steps_per_epoch) if use_ctrl else None

    config = {
        "dataset": dataset, "seed": seed, "method": scheduler_name.upper(),
        "scheduler_name": scheduler_name,
        "scheduler_kwargs": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                              for k, v in scheduler_kwargs.items()},
        "num_epochs": num_epochs,
        "optimizer": ("GALA (own SGD backbone, see gala_config)" if use_gala
                      else "SGD (momentum=0, matches Zaremba et al. 2014)"),
        "model": "Zaremba et al. 2014 medium LSTM (2-layer, non-recurrent-only dropout)",
        "reach_threshold": reach_threshold, "patience": patience, "min_delta": min_delta,
        "ntokens": ntokens, "steps_per_epoch": steps_per_epoch,
        "burnin_epoch_equiv": burnin_epoch_equiv,
        **({"ctrl_" + k: v for k, v in controller_config.items()} if controller_config else {}),
        **({"gala_" + k: v for k, v in gala_config.items()} if gala_config else {}),
        **({"cwgd_" + k: v for k, v in cwgd_cfg.items()} if use_cwgd else {}),
        **{f"recipe_{k}": v for k, v in cfg.items()},
    }
    if extra_config:
        config.update(extra_config)

    run = wandb.init(project=project, name=run_name, config=config, reinit=True)

    reach_epoch = None
    reach_step = None
    final_metric = None  # val perplexity (最終epoch)

    higher_is_better = False  # perplexityは低いほど良い (regressionのRMSEと同じ枠組み)
    best_val_metric = float("inf")
    best_model_state = None
    best_val_epoch = None
    no_improve_count = 0

    ctrl_trace = [] if use_ctrl else None
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        hidden = model.init_hidden(cfg["batch_size"], device)
        for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i, bptt)
            hidden = repackage_hidden(hidden)
            hidden_in = hidden  # ★エントリー時点のhidden state (GALA/CWGDの追加
                                 # forward/backwardで同一ミニバッチとして再利用する)

            model.zero_grad()
            output, hidden = model(data, hidden_in)
            raw_loss = criterion(output.view(-1, ntokens), targets)
            loss = raw_loss
            loss.backward()

            ctrl_info = None
            k_t_pre_clip = None
            if use_ctrl:
                gmag_pre = collect_grad_magnitudes(model, clip_max=None)
                k_t_pre_clip = excess_kurtosis(gmag_pre) if gmag_pre.numel() > 0 else float("nan")

            # Zaremba et al. 2014 medium設定のレシピ: グローバルL2ノルムに基づく勾配クリッピング(clip=5)
            # (要素ごとのクランプではないので kurtosis はこの操作で変化しない。
            #  下のk_t_post_normclipで実データにより経験的に確認する)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])

            cwgd_info = None
            gala_info = None
            if use_ctrl:
                gmag_post = collect_grad_magnitudes(model, clip_max=None)
                k_t_post_normclip = excess_kurtosis(gmag_post) if gmag_post.numel() > 0 else float("nan")
                ctrl_info = controller.update(k_t_pre_clip, optimizer, scheduler=scheduler)
                ctrl_trace.append({
                    "global_step": global_step, "epoch": epoch,
                    "k_t": ctrl_info.get("k_t"), "k_t_post_normclip": k_t_post_normclip,
                    "k_ewm": ctrl_info.get("k_ewm"), "baseline": ctrl_info.get("baseline"),
                    "ratio": ctrl_info.get("ratio"), "current_mult": ctrl_info.get("current_mult"),
                    "action": ctrl_info.get("action"),
                })
            elif use_cwgd:
                # CWGD: batch-first に転置してから update() へ渡す (このモジュール
                # 冒頭のコメント/forward_lstm_for_cwgd_refresh/lstm_per_sample_loss_fn
                # 参照)。
                h0, c0 = hidden_in
                data_bf = data.transpose(0, 1).contiguous()
                h0_bf = h0.transpose(0, 1).contiguous()
                c0_bf = c0.transpose(0, 1).contiguous()
                target_bf = targets.view(data.size(0), data.size(1)).transpose(0, 1).contiguous()
                cwgd_info = cwgd_controller.update(
                    model, criterion, (data_bf, h0_bf, c0_bf), target_bf,
                    optimizer=optimizer, scheduler=scheduler,
                    forward_fn=forward_lstm_for_cwgd_refresh,
                    per_sample_loss_fn=lstm_per_sample_loss_fn,
                )

            if use_gala:
                # GALA は追加逆伝播 (同一 (data, hidden_in) ペアを再利用) →
                # eta_n 算出 → optimizer.step() までを内部で一括して行う。
                # 以下の optimizer.step() は呼ばない。
                gala_info = gala_controller.step(
                    model, (data, hidden_in), targets, criterion, optimizer,
                    forward_fn=forward_lstm_for_gala,
                )
            else:
                optimizer.step()

            if scheduler is not None and is_onecycle:
                scheduler.step()

            if batch_idx % log_every == 0:
                log_dict = {"train/loss": raw_loss.item(), "epoch": epoch,
                            "lr": optimizer.param_groups[0]["lr"]}
                if use_ctrl and ctrl_info is not None:
                    log_dict.update({
                        "grad/kurtosis": ctrl_info.get("k_t", float("nan")),
                        "grad/kurtosis_ewm": ctrl_info.get("k_ewm") or float("nan"),
                        "control/baseline": ctrl_info.get("baseline") or float("nan"),
                        "control/ratio": ctrl_info.get("ratio", float("nan")),
                        "control/current_mult": ctrl_info.get("current_mult", float("nan")),
                        "control/interventions": ctrl_info.get("interventions", 0),
                    })
                if use_gala and gala_info is not None:
                    log_dict.update({
                        "gala/eta": gala_info.get("eta", float("nan")),
                        "gala/L_t": gala_info.get("L_t") if gala_info.get("L_t") is not None else float("nan"),
                        "gala/alignment": gala_info.get("alignment") if gala_info.get("alignment") is not None else float("nan"),
                    })
                if use_cwgd and cwgd_info is not None:
                    log_dict.update({
                        "cwgd/cwgd_t": cwgd_info.get("cwgd_t", float("nan")),
                        "cwgd/mult": cwgd_info.get("mult", float("nan")),
                        "cwgd/refreshed": int(bool(cwgd_info.get("hutchinson_refreshed", False))),
                    })
                wandb.log(log_dict, step=global_step)

            global_step += 1

        val_ppl = evaluate_perplexity(model, val_data, cfg["eval_batch_size"], bptt, device)
        final_metric = val_ppl

        reached = val_ppl <= reach_threshold
        if reach_epoch is None and reached:
            reach_epoch = epoch
            reach_step = global_step - 1
            if use_ctrl and reach_step < controller.burnin_steps:
                print(f"  [WARNING] reach_epoch={epoch} (step={reach_step}) が "
                      f"CTRL burn-in完了(step={controller.burnin_steps})より前です。"
                      f"reach指標がCTRLの介入を反映していない可能性があるため要確認。")

        is_better = val_ppl < best_val_metric - min_delta
        if is_better:
            best_val_metric = val_ppl
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        if use_ctrl and best_val_epoch is not None and \
                (best_val_epoch + 1) * steps_per_epoch < controller.burnin_steps:
            print(f"  [WARNING] best_val_epoch={best_val_epoch} が CTRL burn-in完了"
                  f"(約{burnin_epoch_equiv:.1f}epoch相当)より前です。要確認 (Axis Bで実際に問題になった論点)。")

        if scheduler is not None:
            if is_plateau:
                scheduler.step(val_ppl)
            elif not is_onecycle:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"val/perplexity": val_ppl, "lr": current_lr, "epoch": epoch}, step=global_step - 1)

        if no_improve_count >= patience:
            break

    stop_epoch = epoch

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    final_test_metric = evaluate_perplexity(model, test_data, cfg["eval_batch_size"], bptt, device)
    wandb.log({
        "test/perplexity_final": final_test_metric,
        "train/stop_epoch": stop_epoch, "train/best_val_epoch": best_val_epoch,
        "train/best_val_metric": best_val_metric,
        "epoch": stop_epoch,
    }, step=global_step - 1)

    if use_ctrl and ctrl_trace and save_dir is not None:
        import pandas as pd
        os.makedirs(save_dir, exist_ok=True)
        label = method_label or scheduler_name
        trace_path = os.path.join(save_dir, f"{dataset}_{label}_seed{seed}_ctrl_trace.csv")
        pd.DataFrame(ctrl_trace).to_csv(trace_path, index=False)
        print(f"  [CTRL trace] saved to: {trace_path} "
              f"(burnin_steps={controller_config['burnin_steps']}, "
              f"burnin_epoch_equiv={burnin_epoch_equiv:.2f}, "
              f"baseline={controller.baseline}, final_mult={controller.current_mult:.4f}, "
              f"interventions={controller.interventions})")

    peak_mem_alloc_mb = None
    peak_mem_reserved_mb = None
    if str(device).startswith("cuda"):
        peak_mem_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_mem_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f"  [GPU] peak_memory_allocated={peak_mem_alloc_mb:.1f} MB, "
              f"peak_memory_reserved={peak_mem_reserved_mb:.1f} MB "
              f"(device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        wandb.log({"gpu/peak_memory_allocated_mb": peak_mem_alloc_mb,
                    "gpu/peak_memory_reserved_mb": peak_mem_reserved_mb}, step=global_step - 1)

    run_id = run.id
    wandb.finish()

    return {
        "dataset": dataset, "seed": seed, "run_id": run_id,
        "method": scheduler_name.upper(), "scheduler_name": scheduler_name,
        "reach": 1 if reach_epoch is not None else 0, "reach_threshold": reach_threshold,
        "reach_epoch": reach_epoch, "reach_step": reach_step,
        "final_val_metric": final_metric, "best_val_metric": best_val_metric,
        "best_val_epoch": best_val_epoch, "stop_epoch": stop_epoch,
        "final_test_metric": final_test_metric,
        "peak_memory_allocated_mb": peak_mem_alloc_mb,
        "peak_memory_reserved_mb": peak_mem_reserved_mb,
    }


# =====================================================================
# Multi-seed runner
# =====================================================================

def run_schedulers_multiseed_text(
    dataset: str,
    scheduler_names: List[str],
    *,
    protocol: str = "noes",
    seeds: Tuple[int, ...] = (0, 1, 2),
    device: str = "cuda",
    project: Optional[str] = None,
    num_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    min_delta: Optional[float] = None,
    data_root: str = "/data01/s_hihara/text_data",
    data_source: str = "hf",
    save_dir: Optional[str] = None,
) -> list:
    """dataset × scheduler_names × seeds を実行する。

    Args:
        dataset: "ptb" / "wikitext2"
        scheduler_names: ["ctrl", "cosine", "onecycle", "plateau", "step", "warmup_cosine"]
        protocol: "noes" (early stop 実質なし) / "es2" (patience=20固定、min_deltaは
                  perplexityスケール向けの暫定値。画像・表データ実験と同じpatience=20
                  だが、min_deltaはTEXT_DATASET_CONFIGSのes2_min_deltaを使う点に注意
                  — accuracy/RMSEスケール向けの0.001をそのまま使うのは不適切なため)
    """
    if dataset not in TEXT_DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. choices={list(TEXT_DATASET_CONFIGS)}")
    cfg = TEXT_DATASET_CONFIGS[dataset]

    num_epochs = num_epochs or cfg["num_epochs"]

    if protocol == "noes":
        patience = patience if patience is not None else (num_epochs + 1)
        min_delta = min_delta if min_delta is not None else 0.0
    elif protocol == "es2":
        patience = patience if patience is not None else 20
        min_delta = min_delta if min_delta is not None else cfg["es2_min_delta"]
    else:
        raise ValueError(f"Unknown protocol: {protocol} (choices: noes, es2)")

    project = project or f"KurtosisEWMController-text-{dataset}-{protocol}"

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    print(f"[{dataset}] コーパス読み込み中 (data_source={data_source})...")
    corpus = load_corpus(dataset, data_root, data_source=data_source)
    print(f"[{dataset}] vocab_size={len(corpus.dictionary)}, "
          f"train_tokens={len(corpus.train)}, valid_tokens={len(corpus.valid)}, "
          f"test_tokens={len(corpus.test)}, protocol={protocol} "
          f"(patience={patience}, min_delta={min_delta}), data_source={data_source}")

    # train_data.size(0)(batchify後の長さ)は len(corpus.train)//batch_size と厳密に
    # 一致する (batchifyの定義そのもの) ので、これは近似ではなく厳密値。
    # OneCycleLR等のtotal_stepsはこの値と学習ループの実反復回数を完全一致させる
    # 必要があるため、train_one_seed_text側と同じ steps_per_epoch_from_batchified_len
    # を使う (教訓1: 動的パラメータの不一致を作らない)。
    train_nbatch = len(corpus.train) // cfg["batch_size"]
    steps_per_epoch_est = max(1, steps_per_epoch_from_batchified_len(train_nbatch, cfg["bptt"]))
    total_steps = num_epochs * steps_per_epoch_est
    burnin_epoch_equiv = CTRL_CONFIG["burnin_steps"] / steps_per_epoch_est
    print(f"[{dataset}] steps_per_epoch={steps_per_epoch_est} (厳密値), "
          f"CTRL burn-in({CTRL_CONFIG['burnin_steps']}step) ≈ {burnin_epoch_equiv:.2f} epoch相当")

    all_results = []
    for scheduler_name in scheduler_names:
        lname = scheduler_name.lower()
        is_ctrl = lname == "ctrl"
        is_gala = lname == "gala"
        is_cwgd = lname == "cwgd"

        if is_ctrl:
            effective_scheduler = "cosine"
            effective_kwargs = build_effective_scheduler_kwargs(effective_scheduler, num_epochs, total_steps)
            ctrl_cfg, gala_cfg, cwgd_cfg_local = CTRL_CONFIG, None, None
        elif is_gala:
            # GALA はこの軸自身の CTRL/baseline optimizer 設定 (momentum=0,
            # weight_decay=cfg["wdecay"], base_lr=cfg["base_lr"]) にそのまま
            # 相乗りさせる (train_text_lstm.py 冒頭のコメント参照)。nesterov は
            # momentum=0 では PyTorch SGD が ValueError を出すため False 固定。
            # grad_clip=cfg["clip"] で g_prev にも Zaremba et al. 2014 の
            # global-norm クリッピングを揃える。
            effective_scheduler, effective_kwargs = "gala", {}
            ctrl_cfg, cwgd_cfg_local = None, None
            gala_cfg = dict(
                eta0=cfg["base_lr"], momentum=0.0, nesterov=False,
                weight_decay=cfg["wdecay"], grad_clip=cfg["clip"],
            )
        elif is_cwgd:
            effective_scheduler = "cosine"
            effective_kwargs = build_effective_scheduler_kwargs(effective_scheduler, num_epochs, total_steps)
            ctrl_cfg, gala_cfg = None, None
            cwgd_cfg_local = CWGD_CONFIG
        else:
            effective_scheduler = scheduler_name
            effective_kwargs = build_effective_scheduler_kwargs(effective_scheduler, num_epochs, total_steps)
            ctrl_cfg = gala_cfg = cwgd_cfg_local = None

        for seed in seeds:
            print(f"\n========== [{dataset}] seed={seed}: {scheduler_name.upper()} ==========")
            r = train_one_seed_text(
                dataset=dataset, corpus=corpus, device=device, seed=seed,
                run_name=f"{dataset}-lstm-{protocol}-{scheduler_name}-seed{seed}",
                project=project,
                scheduler_name=effective_scheduler, scheduler_kwargs=effective_kwargs,
                num_epochs=num_epochs, reach_threshold=cfg["reach_threshold"],
                patience=patience, min_delta=min_delta,
                extra_config={"protocol": protocol, "data_source": data_source},
                controller_config=ctrl_cfg, gala_config=gala_cfg, cwgd_config=cwgd_cfg_local,
                save_dir=save_dir, method_label=scheduler_name,
            )
            if is_ctrl:
                r["method"] = "CTRL"
            elif is_gala:
                r["method"] = "GALA"
            elif is_cwgd:
                r["method"] = "CWGD"
            all_results.append(r)
            test_str = (f", final_test_ppl={r['final_test_metric']:.2f}"
                        if r.get("final_test_metric") is not None else "")
            print(f"  -> reach={r['reach']} reach_epoch={r['reach_epoch']} "
                  f"final_val_ppl={r['final_val_metric']:.2f}{test_str}")

            if save_dir is not None:
                import pandas as pd
                pd.DataFrame(all_results).to_csv(
                    os.path.join(save_dir, f"{dataset}_{protocol}_results.csv"), index=False
                )

    return all_results


# =====================================================================
# CLI エントリーポイント (tmux / コマンドライン実行用)
# =====================================================================

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Text (Zaremba et al. 2014 medium LSTM) scheduler comparison (CTRL v5 generality check, Axis C)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(TEXT_DATASET_CONFIGS))
    parser.add_argument("--protocol", type=str, default="noes", choices=["noes", "es2"])
    parser.add_argument(
        "--schedulers", type=str, nargs="+",
        default=["ctrl", "cosine", "onecycle", "plateau", "step", "warmup_cosine"],
        choices=["ctrl", "cosine", "onecycle", "plateau", "step", "warmup_cosine", "gala", "cwgd"],
        help="'gala': SGD-GALA baseline (1 step = 2 backward, ~2x compute)。"
             "'cwgd': CWGD-Cosine baseline — ★torch.func.vmap は LSTM の"
             "バッチングルール未実装のため遅いフォールバック経路を使う"
             "(画像/表データ軸より計算コストが悪化する可能性が高い)。"
             "原論文も非凸実データでの2〜14%性能悪化を明記しているため、"
             "必ず単一seed・小epochのパイロットで先に確認してから"
             "本実験(--seeds 0 1 2 --epochs <本番値>)に進むこと。",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--cuda_visible_devices", type=str, default=None,
        help="例: '0' か '1'。指定した場合、シェル側のCUDA_VISIBLE_DEVICES設定より"
             "優先してこの値を使う。省略時はシェルの環境変数、それも無ければ"
             f"DEFAULT_CUDA_VISIBLE_DEVICES('{DEFAULT_CUDA_VISIBLE_DEVICES}')を使用。"
             "★実行前にAxis A(GPU0)/Axis B(GPU1)の利用状況をユーザーに必ず確認すること。",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="None なら TEXT_DATASET_CONFIGS の既定 epoch 数を使用 (★暫定値)")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--data_root", type=str, default="/data01/s_hihara/text_data")
    parser.add_argument("--data_source", type=str, default="hf", choices=["hf", "local"],
                        help="hf: HuggingFace datasets経由 (既定)。local: 原著getdata.sh相当の"
                             "train.txt/valid.txt/test.txtをdata_root以下から探して読み込む "
                             "(hfが失敗した場合のフォールバック)")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    print(f"dataset={args.dataset}, protocol={args.protocol}, schedulers={args.schedulers}, "
          f"seeds={args.seeds}, device={args.device}, epochs={args.epochs}, "
          f"data_source={args.data_source}")

    results = run_schedulers_multiseed_text(
        dataset=args.dataset, scheduler_names=args.schedulers, protocol=args.protocol,
        seeds=tuple(args.seeds), device=args.device, project=args.project,
        num_epochs=args.epochs, patience=args.patience, min_delta=args.min_delta,
        data_root=args.data_root, data_source=args.data_source, save_dir=args.save_dir,
    )

    df = pd.DataFrame(results)
    print("\n===== Results summary =====")
    summary = df.groupby("method").agg(
        n=("seed", "count"),
        reach_rate=("reach", "mean"),
        reach_epoch_mean=("reach_epoch", "mean"),
        reach_epoch_std=("reach_epoch", "std"),
        stop_epoch_mean=("stop_epoch", "mean"),
        stop_epoch_std=("stop_epoch", "std"),
        test_ppl_mean=("final_test_metric", "mean"),
        test_ppl_std=("final_test_metric", "std"),
    )
    print(summary)
