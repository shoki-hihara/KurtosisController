import os

os.environ["TMPDIR"]             = "/data01/s_hihara/tmp"
os.environ["WANDB_DIR"]          = "/data01/s_hihara/wandb"
os.environ["WANDB_CACHE_DIR"]    = "/data01/s_hihara/wandb/cache"

import wandb
# W&B は事前に `wandb login` で認証しておくこと
# wandb.login(key="YOUR_API_KEY")  ← キーはコードに書かず環境変数 WANDB_API_KEY を使う

import numpy
import torch

# =====================================================================
# GPU 選択 (★ 2026-08-23 変更: 実行コード側から切り替え可能にする)
#
# 優先順位: (1) --cuda_visible_devices CLI引数 (__main__ 内で明示的に上書き)
#         > (2) シェルで事前に設定された CUDA_VISIBLE_DEVICES 環境変数
#         > (3) 下記 DEFAULT_CUDA_VISIBLE_DEVICES (このファイルにしか無い場合の
#               フォールバック)
#
# 旧実装は os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_... で無条件上書き
# していたため、`CUDA_VISIBLE_DEVICES=1 python train_....py` のように外側
# から指定しても常に "0" に潰されてしまっていた。setdefault() に変えることで
# 外側の指定を尊重しつつ、未指定時のみこのデフォルト値を使う。
# GPUを2枚使える状況だが「他の人が使いたければすぐ譲る」運用のため、
# 実行のたびに --cuda_visible_devices "0" / "1" で明示的に切り替えられる
# ようにしてある (使い方は下のdocstring参照)。
# =====================================================================

DEFAULT_CUDA_VISIBLE_DEVICES = "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_CUDA_VISIBLE_DEVICES)

"""
train_tabular_sgd_diag.py — CTRL v6 診断実験専用フォーク
(このファイルは train_tabular_ft_transformer.py [2026-08-29時点] を複製したもの。
Axis B本実験のファイルとは別物であり、本実験の結果には一切使わない。2026-08-30、
v6設計検討専用チャットにて、「CTRLが機能するのはSGD前提だからではないか」という
仮説を検証するために作成した。変更点は optimizer_override 関連の追加のみで、CTRL
自体 (ContinuousStateKurtosisController)・CTRL_CONFIG は一切変更していない。詳細は
project_ctrl_v6_design.md および同フォルダの README.md 参照。)

--- 以下、元ファイル train_tabular_ft_transformer.py の docstring (Axis B用の説明。
このファイルの実行には直接関係しない箇所を含むが、CTRL自体の設計方針・共通の
確定事項は変更なくそのまま適用されるため、参考として残してある) ---

train_tabular_ft_transformer.py — CTRL v5 汎用性検証（表データ, Axis B）

============================================================================
目的
============================================================================
CTRL v5 (ContinuousStateKurtosisController) が画像以外のモダリティ（表データ）
でも同じハイパラのまま機能するかを検証する（汎用性の主張の根拠）。
画像データセット実験 (train_v5_multidataset_schedulers.py) とは別チャットで
並行して進めている実験。

対象データセット: Adult (二値分類) / California Housing (回帰) / Covertype (多クラス分類)
モデル          : FT-Transformer (Gorishniy et al. 2021)
比較手法        : CTRL / COSINE / WARMUP_COSINE / ONECYCLE / PLATEAU / STEP

位置づけ (★ 2026-08-23): 現在GPUはImageNet実験(画像側チャット)で使用中のため、
本スクリプトによる表データ実験は「本番実験に向けた調整用実験」という位置づけ。
ImageNet実験が終わり次第、本スクリプトで本番実験(複数seed本実行)に入る。

このセッションでの検証状況 (★重要): GPU/実データ無しのCowork環境上で、
sklearn_fallbackデータソース + 合成データ (ランダム生成、公式splitではない) を使い、
3データセット×6スケジューラ (CTRL含む) すべてでCPU上でのend-to-end動作確認
(前処理→モデル構築→CTRL適用込みの訓練ループ→reach判定→best checkpoint選定→
test-once評価→W&Bログ→CSV集計) を実施し、構文・ロジックの両面でエラーなく
完走することを確認済み。一方で (1) data.tar.gz の実際のディレクトリ構成、
(2) 実GPUでのVRAM使用量・1epochあたりの実測時間、(3) 実データでの収束挙動・
kurtosisの推移 (burn-in直後のspike有無など) は未検証。ラボサーバでの
最初のpilot実行で必ず確認すること。

============================================================================
設計方針（★ 2026-08-23 確定、project_tabular_experiments.md を継承）
============================================================================

【CTRL ハイパラは全データセット共通・再チューニングしない】
    alpha=0.05, burnin_steps=2000, alpha_decay=0.1, beta=0.99, min_mult=0.2
    （画像実験で確定した値をそのまま使用。ContinuousStateKurtosisController
    クラス自体は一切変更しない。）

【burn-in整合性の注意（★要pilot確認）】
    burnin_steps=2000 は step 数ベースなので、step/epoch が小さいデータセットほど
    burn-in に要する epoch 数が大きくなる。概算:
      - Adult            : train 26,048 / batch 256 ≈ 102 step/epoch → burn-in完了に約20epoch
      - California Housing: train 13,209 / batch 256 ≈  52 step/epoch → burn-in完了に約38epoch (要注意)
      - Covertype        : train 371,847 / batch 1024 ≈ 363 step/epoch → burn-in完了に約5.5epoch (画像実験と同程度)
    California Housing は訓練全体に占める burn-in の割合が大きくなるリスクがある。
    ImageNet-100 で見られた「burn-in直後のspike→monotone_decreaseで永久にLDが下がる」
    over-damping問題が起きていないか、pilot実行時に最優先で確認すること。
    起きていてもハイパラは再チューニングせず、限界として正直に報告する方針。

【FT-Transformer実装: rtdl_revisiting_models (公式ライブラリ) を使用】
    pip install rtdl_revisiting_models
    FTTransformer.get_default_kwargs() + model.make_default_optimizer() で
    チューニングなしのデフォルト構成 (AdamW, n_blocks=3, d_block=192 等) を使う。
    FT-Transformer自体のハイパラ (アーキテクチャ・optimizerのlr/wd) は
    ライブラリのデフォルトにすべて委ね、こちらで独自にチューニングしない。
    ★ 2026-08-23、Cowork環境 (CPU, rtdl_revisiting_models最新版) で
    get_default_kwargs()/FTTransformer(...)/make_default_optimizer()/forward の
    呼び出し方をすべて実際に動かして確認済み (n_cont_features=0や
    cat_cardinalities=[]のケース、d_out=1/7の各ケースを含む)。
    get_default_kwargs()はn_blocks=3のデフォルト1引数のみで呼び出し可能、
    make_default_optimizer()はAdamW(lr=1e-4, weight_decay=1e-5、
    tokenizer/LayerNorm/bias相当の第2パラメータグループはweight_decay=0)を返す
    ことを確認した。ラボサーバのrtdl_revisiting_modelsバージョンが極端に古い/新しい
    場合のみ念のため再確認すること。

【データ分割: FT-Transformer原論文の公式splitを採用 (★2026-08-23確定)】
    画像実験の split_seed=20260527 による独自再分割はここでは行わない。
    3データセットとも原論文 (Gorishniy et al. 2021, Table 7) が使う「正確に1つの
    公式train/val/testスプリット」が存在し、多くのtabular DL論文で再利用されて
    いる定番のため、それをそのまま採用する方が「独自ベンチマークを作らない」
    方針に忠実。

    入手方法:
        wget "https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1" \
            -O revisiting_models_data.tar.gz
        tar -xvf revisiting_models_data.tar.gz -C <data_root>
    展開後、<data_root>/adult/ , <data_root>/california_housing/ , <data_root>/covtype/
    のようなデータセットごとのディレクトリに、N_train.npy (数値特徴量) /
    C_train.npy (カテゴリカル特徴量, 存在する場合のみ) / y_train.npy (ターゲット) /
    info.json 等が格納されている想定 (yandex-research系リポジトリの標準フォーマット)。
    ★ このフォーマットの詳細は未検証 (このセッションにはネットワークアクセス+実機が
    無くdata.tar.gzの中身を直接確認できていない)。ラボサーバでの初回pilot実行時に
    load_official_tabular_split() が失敗したら、エラーメッセージに従い実際の
    ディレクトリ構成 (ls -la <data_root>/<dataset>/) を確認し、ファイル名を合わせて
    修正すること。

    前処理は原論文 B.2節の記述に合わせる:
      - 数値特徴量: sklearn QuantileTransformer(output_distribution="normal") を
        train集合のみでfitし、val/testに適用。fit前に数値安定化のため
        N(0, 1e-3) の微小ノイズを加える (原論文と同じ処理)。
      - カテゴリカル特徴量: 整数インデックスにラベルエンコード (trainでfit、
        val/testの未知カテゴリは追加の「unknownバケット」に割り当てる)。
      - 回帰(California Housing)のターゲット: train集合のmean/stdで標準化して
        学習。評価時 (RMSE) は元のスケールに逆変換してから計算する。

    もし何らかの理由でdata.tar.gzが入手できない場合のフォールバックとして
    --data_source sklearn_fallback を用意した (fetch_openml/fetch_california_housing/
    fetch_covtype から about 80/10/10 で再分割する)。★これは原論文の公式splitとは
    一致しないため、パイプラインの動作確認用途に限定し、本実験では使わないこと。

【FS/TLの区別はしない・評価プロトコルはnoESとES2の両方 (画像実験と同一方針)】
    --protocol noes : patience=epochs+1 (実質的に early stop しない), min_delta=0.0
    --protocol es2  : patience=20, min_delta=0.001
    ★ このpatience=20は原論文自身のpatience=16とは異なる値。画像実験と同一条件で
    比較できるようにするためにプロジェクト独自に定めたES2定義をそのまま踏襲して
    いるので、原論文の値に合わせているわけではない (意図的)。

【タスクごとの損失関数・評価指標】
      - Adult (binclass)      : BCEWithLogitsLoss, d_out=1, 指標=accuracy (閾値0.5)
      - Covertype (multiclass): CrossEntropyLoss,   d_out=7, 指標=accuracy
      - California Housing (regression): MSELoss (標準化後のyに対して), d_out=1,
        指標=RMSE (元のスケールに逆変換してから計算)。reach_threshold の向きが
        分類 (>=) と回帰 (<=、低いほど良い) で逆になる点に注意 (TASK_METRIC_HIGHER_IS_BETTER
        で分岐)。

【reach_threshold ("reach75相当") と num_epochs は現時点では暫定値 (★要pilot後確定)】
    DATASET_CONFIGS の reach_threshold / num_epochs はいずれも暫定値であり、
    pilot実行の結果を見てから正式に決定する方針 (ImageNet-100のreach_threshold=0.80
    と同じ扱い)。num_epochs=100 (Covertypeも含め暫定的に統一) としているが、原論文は
    epoch上限を明記せずpatience=16のみで学習を止めているため、根拠のある値ではない。
    pilotで収束の速さを見て、短すぎ/長すぎであれば調整すること。

============================================================================
使い方 (CLI)
============================================================================
    # Adult, noES プロトコル, 全スケジューラ, 5 seed, GPU0を使用
    python train_tabular_ft_transformer.py --dataset adult --protocol noes \
        --seeds 0 1 2 3 4 --device cuda --cuda_visible_devices 0 \
        --data_root /data01/s_hihara/tabular_data --save_dir ./results_adult_noes

    # 同じ実験をGPU1で実行したい場合 (他の人にGPU0を譲る場合など)
    python train_tabular_ft_transformer.py --dataset adult --protocol noes \
        --seeds 0 1 2 3 4 --device cuda --cuda_visible_devices 1 \
        --data_root /data01/s_hihara/tabular_data --save_dir ./results_adult_noes

    # --cuda_visible_devices を省略した場合は、シェル側で
    #   CUDA_VISIBLE_DEVICES=1 python train_tabular_ft_transformer.py ...
    # のように指定しても (スクリプト側で上書きされず) そちらが尊重される。
    # どちらも指定しなければ DEFAULT_CUDA_VISIBLE_DEVICES ("0") が使われる。

    # California Housing, ES2 プロトコル
    python train_tabular_ft_transformer.py --dataset california_housing --protocol es2 \
        --seeds 0 1 2 3 4 --device cuda --data_root /data01/s_hihara/tabular_data \
        --save_dir ./results_california_housing_es2

    # Covertype
    python train_tabular_ft_transformer.py --dataset covtype --protocol noes \
        --seeds 0 1 2 3 4 --device cuda --data_root /data01/s_hihara/tabular_data \
        --save_dir ./results_covtype_noes

    # 動作確認だけしたい場合（1 seed, 数 epoch, cosineのみ, "調整用"実験の第一歩）
    python train_tabular_ft_transformer.py --dataset adult --protocol noes \
        --schedulers cosine --seeds 0 --epochs 3 --device cuda

    # data.tar.gzが手元にない場合のパイプライン動作確認 (公式splitではない点に注意)
    python train_tabular_ft_transformer.py --dataset adult --protocol noes \
        --schedulers cosine --seeds 0 --epochs 2 --device cuda --data_source sklearn_fallback
"""

import json
import math
import random
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call

# GALA / CWGD ベースライン (2026-08-29 追加、[[project_ctrl_related_work]] 優先度1/2)
from gala_optimizer import GALAController
from cwgd_controller import CWGDController


# =====================================================================
# Kurtosis / CTRL controller
# (train_v5_multidataset_schedulers.py と完全同一。データセット・モダリティに
#  依存しないコアロジックなのでそのまま流用する。一切変更しない。)
# =====================================================================

@torch.no_grad()
def excess_kurtosis(x: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.float()
    if x.numel() < 10:
        return float("nan")
    mu = x.mean()
    v = x.var(unbiased=False)
    if v < eps:
        return 0.0
    m4 = ((x - mu) ** 4).mean()
    k = m4 / (v ** 2 + eps) - 3.0
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


# CTRL ハイパラ: 全データセット共通・再チューニングなし (★確定値、画像実験と完全同一)
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
# GALA / CWGD ハイパラプリセット (2026-08-29 追加、優先度1/2 ベースライン、
# [[project_ctrl_related_work]])
# =====================================================================
#
# ★★★ optimizer backbone に関する重要な設計判断 ★★★
# 他の全 method (CTRL 含む) はこのファイルでは `model.make_default_optimizer()`
# (rtdl_revisiting_models の FTTransformer 用 AdamW ベース、パラメータ群ごとに
# weight_decay が異なる) を共通の optimizer backbone として使っている。
#
# CWGD-Cosine は CTRL と同じく「既存の CosineAnnealingLR の base_lrs を変調する」
# だけなので、この AdamW backbone にそのまま相乗りできる (画像実験と同じ
# フェアネスを保てる)。
#
# 一方 GALA (SGD-GALA) は Lipschitz 推定・alignment 計算が
# 「x_{t+1} = x_t - eta_t * g_t(x_t)」という (momentum 付き) SGD 型の更新則を
# 前提にしており、Adam 系の per-parameter 適応的更新には理論的に対応しない
# (原論文・引継ぎ仕様書とも "SGD-GALA" としか定義していない)。そのため GALA は
# tabular 軸でも `model.make_default_optimizer()` を使わず、
# GALAController.build_optimizer() が作る SGD(momentum) を専用の optimizer
# backbone として使う。
#
# → この結果、GALA と他手法 (CTRL/COSINE/...) の比較は「同一 optimizer
#   backbone 上での LR 制御機構の比較」ではなく「異なる最適化パラダイム間の
#   比較」になる。論文・報告書では必ずこの限界を明記すること。
#
# eta0 (初期学習率) は AdamW の既定 lr (rtdl 既定は 1e-4 台) とスケールが全く
# 異なる SGD 用の値なので、他軸 (画像: base_lr=0.1) の慣例に合わせて 0.1 を
# 既定値としているが、★ 表データでこの値が妥当かは未検証。必ずパイロットで
# 発散しないか確認し、必要なら調整すること (国際学会方針一覧の「非再チューニング」
# 原則は「調整後の値を全データセット・全 seed で固定する」ことを指し、
# 「最初から動くとわかっている値を使う」ことまでは免除しない)。
GALA_TABULAR_CONFIG = dict(
    eta0=0.1,
    momentum=0.9,
    nesterov=True,
    weight_decay=0.0,
)

# ★CTRL v6診断実験専用 (2026-08-30追加、このファイルにのみ存在)。
# 「CTRLが機能するのはSGD前提だからではないか」という仮説を検証するため、
# --optimizer_override sgd 指定時のみ、AdamW backbone (model.make_default_optimizer())
# の代わりにこの SGD backbone を使う。値は GALA_TABULAR_CONFIG (このファイル上部) と
# 同一のものを流用している(このFT-Transformer+SGDの組み合わせで既に使われている
# 前例があるため。ただしGALA_TABULAR_CONFIG自身のコメントにある通り、この lr=0.1 が
# 表データで発散しないかは要pilot確認、という限界はこちらにもそのまま当てはまる)。
# CTRL自体のハイパラ (CTRL_CONFIG) は一切変更しない。
SGD_DIAG_TABULAR_CONFIG = dict(
    lr=0.1,
    momentum=0.9,
    nesterov=True,
    weight_decay=0.0,
)

# CWGD: alpha=1.0 は原論文推奨値。refresh_interval_steps=None は
# train_one_seed_tabular 内で Delta≈T/8 epoch 相当を自動導出する合図。
# ★ 原論文は非凸実データでの性能悪化を明記しているため、必ず単一 seed・
# 少 epoch のパイロットで COSINE に対して悪化しないか確認してから
# 本実験に進むこと ([[project_ctrl_related_work]] / cwgd_controller.py 参照)。
CWGD_CONFIG = dict(
    alpha=1.0,
    num_probes=20,
    hutchinson_delta=1e-3,
    refresh_interval_steps=None,
    lambda_floor=1e-6,
    subsample_size=None,
)


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
# データ: 公式split読み込み・前処理
# =====================================================================

DEFAULT_DATA_ROOT = "/data01/s_hihara/tabular_data"

# 原論文 (Gorishniy et al. 2021, Table 7) の公式split行数。読み込んだデータが
# これと一致するかをアサートし、フォーマット取り違えを早期に検知する。
OFFICIAL_SPLIT_SIZES = {
    "adult":               dict(train=26_048, val=6_513,  test=16_281),
    "california_housing":  dict(train=13_209, val=3_303,  test=4_128),
    "covtype":             dict(train=371_847, val=92_962, test=116_203),
}

# yandex-research 系リポジトリ (rtdl-revisiting-models 等) が配布する
# data.tar.gz 内のディレクトリ名候補。tar.gz が1階層ラップされている場合や、
# 略称/別名で配布されている場合に備えて複数の候補を持たせ、_resolve_dataset_dir()
# で data_root 以下を再帰的に探索してマッチさせる (★2026-08-24追加: 直下に固定
# パスがある前提のOFFICIAL_DATASET_DIRNAME方式では実際のラボサーバで見つからな
# かったため、再帰探索+候補名マッチ方式に変更)。
OFFICIAL_DATASET_DIRNAME_CANDIDATES = {
    "adult": ["adult", "adult_income", "census_income", "census"],
    "california_housing": [
        "california_housing", "california-housing", "californiahousing",
        "cal_housing", "calhousing", "ca_housing", "housing",
    ],
    "covtype": ["covtype", "covertype", "cover_type", "forest_cover", "forestcover"],
}


def _load_split_npy(data_dir: str, prefix: str, split: str) -> Optional[np.ndarray]:
    """<data_dir>/<prefix>_<split>.npy があれば読み込み、無ければ None を返す
    (例: N_train.npy が無いデータセット = 数値特徴量が無い、等のケースに対応)。"""
    path = os.path.join(data_dir, f"{prefix}_{split}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=(prefix == "C"))


def _resolve_dataset_dir(dataset: str, data_root: str) -> Optional[str]:
    """data_root 以下を再帰的に探索し、dataset に該当するディレクトリを探す。
    tar.gz展開後に1階層余分にラップされているケース (例: <data_root>/data/adult/)
    や、フォルダ名が想定と多少違うケース (大文字小文字・ハイフン等) に対応するため、
    直下だけでなく再帰的に探索し、候補名リストと大小文字を無視して比較する。
    N_train.npy/C_train.npy/y_train.npyのいずれかを実際に含むディレクトリのみを
    候補とする (単なる同名ディレクトリの誤マッチを避けるため)。
    """
    candidates = [c.lower() for c in OFFICIAL_DATASET_DIRNAME_CANDIDATES[dataset]]
    matches = []
    for root, dirs, files in os.walk(data_root):
        base = os.path.basename(root).lower()
        if base in candidates:
            has_data_file = any(
                f.startswith(("N_train", "C_train", "y_train")) for f in files
            )
            if has_data_file:
                matches.append(root)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # 複数ヒットした場合は、行数がTable 7の想定値に一致するものを優先する
        expected_train = OFFICIAL_SPLIT_SIZES[dataset]["train"]
        for m in matches:
            y = _load_split_npy(m, "y", "train")
            if y is not None and len(y) == expected_train:
                return m
        return matches[0]  # 一致するものが無ければ最初の候補を返す (呼び出し元で行数チェックされる)
    return None


def _list_data_root_tree(data_root: str, max_lines: int = 200) -> str:
    """診断用: data_root 以下のディレクトリ・ファイルを一覧化する
    (自動探索に失敗した場合、エラーメッセージに含めてユーザーに実際の構成を
    伝えるため)。"""
    lines = []
    for root, dirs, files in os.walk(data_root):
        depth = root[len(data_root):].count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        indent = "  " * depth
        lines.append(f"{indent}{os.path.basename(root) or root}/")
        for f in sorted(files)[:10]:
            lines.append(f"{indent}  {f}")
        if len(files) > 10:
            lines.append(f"{indent}  ... (他{len(files) - 10}ファイル)")
        if len(lines) >= max_lines:
            lines.append("... (以下省略)")
            break
    return "\n".join(lines)


def load_official_tabular_split(dataset: str, data_root: str) -> dict:
    """FT-Transformer原論文が配布する公式 train/val/test split を読み込む。

    想定ディレクトリ構成:
        <data_root>/(...任意の階層.../)<dataset相当のフォルダ>/
            N_train.npy, N_val.npy, N_test.npy   # 数値特徴量 (float)。無い場合は省略可
            C_train.npy, C_val.npy, C_test.npy   # カテゴリカル特徴量 (object/str)。無い場合は省略可
            y_train.npy, y_val.npy, y_test.npy   # ターゲット
            info.json                             # {"task_type": "binclass"/"multiclass"/"regression", ...}

    ★2026-08-24: 直下の固定パスを仮定せず、data_root以下を再帰的に探索して
    データセットフォルダを自動検出する方式に変更 (_resolve_dataset_dir参照)。
    それでも見つからない場合は、data_root以下の実際の構成をエラーメッセージに
    含めて表示する。

    入手方法 (ラボサーバ上で一度だけ実行):
        wget "https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1" \\
            -O revisiting_models_data.tar.gz
        mkdir -p <data_root> && tar -xvf revisiting_models_data.tar.gz -C <data_root>
    """
    data_dir = _resolve_dataset_dir(dataset, data_root)

    if data_dir is None:
        tree = _list_data_root_tree(data_root) if os.path.isdir(data_root) else "(data_root自体が存在しません)"
        raise FileNotFoundError(
            f"公式データ ({dataset}) が {data_root} 以下に見つかりませんでした。\n"
            f"候補名 {OFFICIAL_DATASET_DIRNAME_CANDIDATES[dataset]} のいずれとも一致する、"
            f"N_train.npy/C_train.npy/y_train.npyを含むディレクトリが見つかりません。\n\n"
            f"=== {data_root} 以下の実際の構成 ===\n{tree}\n\n"
            f"まだダウンロードしていない場合は以下を実行してください:\n"
            f'  wget "https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1" '
            f"-O revisiting_models_data.tar.gz\n"
            f"  mkdir -p {data_root} && tar -xvf revisiting_models_data.tar.gz -C {data_root}\n"
            f"上の「実際の構成」がすでに何かデータらしきものを含んでいる場合は、その内容を"
            f"Claudeに伝えてください (OFFICIAL_DATASET_DIRNAME_CANDIDATES / _load_split_npy の"
            f"ファイル名パターンを実際の形式に合わせて修正します)。\n"
            f"(--data_source sklearn_fallback で公式split以外のデータでパイプライン動作確認も可能)"
        )

    result = {}
    for split in ("train", "val", "test"):
        N = _load_split_npy(data_dir, "N", split)
        C = _load_split_npy(data_dir, "C", split)
        y = _load_split_npy(data_dir, "y", split)
        if y is None:
            existing = os.listdir(data_dir)
            raise FileNotFoundError(
                f"{data_dir} に y_{split}.npy が見つかりません。\n"
                f"実際のディレクトリ内容: {existing}\n"
                f"想定していたファイル名パターン (N_/C_/y_<split>.npy) と実際の形式が"
                f"異なる可能性があります。ファイル名を確認しload_official_tabular_split()を修正してください。"
            )
        result[split] = dict(N=N, C=C, y=y)

    info_path = os.path.join(data_dir, "info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
    result["info"] = info

    expected = OFFICIAL_SPLIT_SIZES[dataset]
    actual = {s: len(result[s]["y"]) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        if actual[s] != expected[s]:
            print(
                f"[WARN] {dataset}/{s}: 行数が原論文Table 7の想定値と一致しません "
                f"(actual={actual[s]}, expected={expected[s]})。データソースが公式splitと"
                f"異なる可能性があるため、本実験に使う前に確認してください。"
            )

    return result


def load_sklearn_fallback_split(dataset: str, split_seed: int = 20260527) -> dict:
    """data.tar.gzが入手できない場合のフォールバック。sklearnの標準取得関数から
    80/10/10でtrain/val/testに再分割する。★原論文の公式splitとは一致しないので
    パイプラインの動作確認 (--data_source sklearn_fallback) にのみ使うこと。"""
    from sklearn.model_selection import train_test_split

    if dataset == "adult":
        from sklearn.datasets import fetch_openml
        data = fetch_openml("adult", version=2, as_frame=True)
        df = data.frame.dropna()
        y = (df["class"].astype(str).str.strip() == ">50K").astype(int).to_numpy()
        df = df.drop(columns=["class"])
        cat_cols = [c for c in df.columns if df[c].dtype.name in ("category", "object")]
        num_cols = [c for c in df.columns if c not in cat_cols]
        N = df[num_cols].to_numpy(dtype=np.float32) if num_cols else None
        C = df[cat_cols].astype(str).to_numpy() if cat_cols else None
        task_type = "binclass"
    elif dataset == "california_housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        N = data.data.astype(np.float32)
        C = None
        y = data.target.astype(np.float32)
        task_type = "regression"
    elif dataset == "covtype":
        from sklearn.datasets import fetch_covtype
        data = fetch_covtype()
        N = data.data.astype(np.float32)
        C = None
        y = (data.target - 1).astype(np.int64)  # 1-7 -> 0-6
        task_type = "multiclass"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=split_seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=split_seed)

    def _sub(arr, ix):
        return None if arr is None else arr[ix]

    result = {}
    for split, ix in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        result[split] = dict(N=_sub(N, ix), C=_sub(C, ix), y=_sub(y, ix))
    result["info"] = dict(task_type=task_type)
    print(f"[WARN] {dataset}: sklearn_fallback split を使用中 (公式splitではありません)")
    return result


class TabularPreprocessor:
    """原論文 B.2節に合わせた前処理 (train集合のみでfit):
      - 数値特徴量: 微小ノイズ付加 + QuantileTransformer(output_distribution="normal")
      - カテゴリカル特徴量: ラベルエンコード (train未出現カテゴリはunknownバケットへ)
      - 回帰ターゲット: train集合のmean/stdで標準化 (RMSE算出時に逆変換して使う)
    """

    def __init__(self, task_type: str, split_seed: int = 20260527):
        self.task_type = task_type
        self.split_seed = split_seed
        self.qt = None
        self.cat_maps: List[Dict[str, int]] = []
        self.cat_cardinalities: List[int] = []
        self.y_mean = None
        self.y_std = None

    def fit_transform(self, splits: dict) -> dict:
        from sklearn.preprocessing import QuantileTransformer

        out = {s: {} for s in ("train", "val", "test")}

        # --- 数値特徴量 ---
        N_train = splits["train"]["N"]
        if N_train is not None:
            rng = np.random.default_rng(self.split_seed)
            noise = rng.normal(0.0, 1e-3, size=N_train.shape).astype(np.float32)
            n_quantiles = max(10, min(1000, N_train.shape[0]))
            self.qt = QuantileTransformer(
                output_distribution="normal", n_quantiles=n_quantiles,
                subsample=min(1_000_000, N_train.shape[0]), random_state=self.split_seed,
            )
            self.qt.fit(N_train.astype(np.float32) + noise)
            for s in ("train", "val", "test"):
                N = splits[s]["N"]
                out[s]["x_num"] = torch.tensor(
                    self.qt.transform(N.astype(np.float32)), dtype=torch.float32
                ) if N is not None else torch.zeros((len(splits[s]["y"]), 0), dtype=torch.float32)
        else:
            for s in ("train", "val", "test"):
                out[s]["x_num"] = torch.zeros((len(splits[s]["y"]), 0), dtype=torch.float32)

        # --- カテゴリカル特徴量 ---
        C_train = splits["train"]["C"]
        if C_train is not None:
            n_cat = C_train.shape[1]
            self.cat_maps = []
            self.cat_cardinalities = []
            for j in range(n_cat):
                uniques = sorted(set(str(v) for v in C_train[:, j]))
                mapping = {v: i for i, v in enumerate(uniques)}
                # 未知カテゴリ (val/testにのみ出現) 用に +1 して追加バケットを確保する。
                # 実際のインデックス割り当ては下のループで len(mapping) を都度参照する。
                self.cat_maps.append(mapping)
                self.cat_cardinalities.append(len(uniques) + 1)
            for s in ("train", "val", "test"):
                C = splits[s]["C"]
                cols = []
                for j in range(n_cat):
                    mapping = self.cat_maps[j]
                    unknown_idx = len(mapping)
                    col = np.array(
                        [mapping.get(str(v), unknown_idx) for v in C[:, j]], dtype=np.int64
                    )
                    cols.append(col)
                out[s]["x_cat"] = torch.tensor(np.stack(cols, axis=1), dtype=torch.long)
        else:
            self.cat_cardinalities = []
            for s in ("train", "val", "test"):
                out[s]["x_cat"] = torch.zeros((len(splits[s]["y"]), 0), dtype=torch.long)

        # --- ターゲット ---
        y_train = splits["train"]["y"]
        if self.task_type == "regression":
            self.y_mean = float(np.mean(y_train))
            self.y_std = float(np.std(y_train) + 1e-12)
            for s in ("train", "val", "test"):
                y = splits[s]["y"].astype(np.float32)
                out[s]["y"] = torch.tensor((y - self.y_mean) / self.y_std, dtype=torch.float32)
        elif self.task_type == "binclass":
            for s in ("train", "val", "test"):
                out[s]["y"] = torch.tensor(splits[s]["y"].astype(np.float32), dtype=torch.float32)
        else:  # multiclass
            for s in ("train", "val", "test"):
                out[s]["y"] = torch.tensor(splits[s]["y"].astype(np.int64), dtype=torch.long)

        return out

    def denormalize_y(self, y_std_scale: torch.Tensor) -> torch.Tensor:
        """regressionタスクのみ: 標準化されたyを元のスケールに戻す。"""
        assert self.task_type == "regression"
        return y_std_scale * self.y_std + self.y_mean


TABULAR_DATASETS = ("adult", "california_housing", "covtype")

# 分類は accuracy が高いほど良い (reach_threshold は ">="), 回帰の RMSE は
# 低いほど良い (reach_threshold は "<=")。train_one_seed_tabular / reach判定
# / best-checkpoint選定で参照する。task_type (データセットではなく) をキーにする
# ことで、将来別のregressionデータセットが増えても取り違えない。
TASK_METRIC_HIGHER_IS_BETTER = {
    "binclass": True,     # accuracy
    "multiclass": True,   # accuracy
    "regression": False,  # RMSE (低いほど良い)
}


def make_tabular_loaders(
    dataset: str, batch_size: int, data_root: str, data_source: str = "official",
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """公式split (またはsklearn_fallback) を読み込み・前処理し、DataLoaderを返す。

    Returns: (train_loader, val_loader, test_loader, meta)
        meta には task_type / n_cont_features / cat_cardinalities / preprocessor を含む。
    """
    if data_source == "official":
        splits = load_official_tabular_split(dataset, data_root)
    elif data_source == "sklearn_fallback":
        splits = load_sklearn_fallback_split(dataset)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    task_type = splits["info"].get("task_type") or {
        "adult": "binclass", "california_housing": "regression", "covtype": "multiclass",
    }[dataset]

    pre = TabularPreprocessor(task_type=task_type)
    processed = pre.fit_transform(splits)

    loaders = {}
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        ds = TensorDataset(processed[split]["x_num"], processed[split]["x_cat"], processed[split]["y"])
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    meta = dict(
        task_type=task_type,
        n_cont_features=processed["train"]["x_num"].shape[1],
        cat_cardinalities=list(pre.cat_cardinalities),
        preprocessor=pre,
    )
    return loaders["train"], loaders["val"], loaders["test"], meta


# =====================================================================
# モデル: FT-Transformer (rtdl_revisiting_models)
# =====================================================================

def build_ft_transformer(meta: dict):
    """rtdl_revisiting_models の FTTransformer をデフォルト構成 (チューニングなし)
    で構築する。d_out はタスク種別で決まる (binclass=1, multiclass=n_classes, regression=1)。

    ★ get_default_kwargs() の引数要否は未検証。pilot実行時にTypeErrorが出た場合は
    rtdl_revisiting_modelsの実際のバージョンに合わせて呼び出し方を調整すること。
    """
    from rtdl_revisiting_models import FTTransformer

    task_type = meta["task_type"]
    if task_type == "binclass":
        d_out = 1
    elif task_type == "regression":
        d_out = 1
    elif task_type == "multiclass":
        d_out = meta["n_classes"]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    default_kwargs = FTTransformer.get_default_kwargs()
    model = FTTransformer(
        n_cont_features=meta["n_cont_features"],
        cat_cardinalities=meta["cat_cardinalities"],
        d_out=d_out,
        **default_kwargs,
    )
    return model


def forward_ft_transformer(model, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
    x_cat_arg = x_cat if x_cat.shape[1] > 0 else None
    out = model(x_num, x_cat_arg)
    return out.squeeze(-1) if out.ndim == 2 and out.shape[1] == 1 else out


def make_ft_transformer_per_sample_loss_fn(criterion):
    """CWGDController の per_sample_loss_fn として使う関数を組み立てる。

    forward_ft_transformer と同じ「x_cat が空なら None を渡す・出力を squeeze
    する」というロジックを、functional_call (params/buffers 差し替え) 経由でも
    再現する必要があるため、通常の forward_ft_transformer とは別に用意する
    (torch.func.functional_call は model.__call__ を直接呼ぶだけで、
    forward_ft_transformer のような外側のラッパー関数までは経由しないため)。
    criterion をクロージャで束ねているのは、CWGDController.update() の
    per_sample_loss_fn 規約 (model, params, buffers, *sample_args, target) が
    criterion を引数に含まないため。
    """
    def per_sample_loss_fn(model, params, buffers, x_num_1, x_cat_1, target_1):
        x_num_b = x_num_1.unsqueeze(0)
        x_cat_b = x_cat_1.unsqueeze(0)
        x_cat_arg = x_cat_b if x_cat_b.shape[1] > 0 else None
        out = functional_call(model, (params, buffers), (x_num_b, x_cat_arg))
        out = out.squeeze(-1) if out.ndim == 2 and out.shape[1] == 1 else out
        tgt = target_1.unsqueeze(0)
        return criterion(out, tgt)
    return per_sample_loss_fn


# =====================================================================
# スケジューラ (train_v5_multidataset_schedulers.py と同一。optimizerの
# 種類 (SGD/AdamW) に依存しない汎用ロジックなので変更不要)
# =====================================================================

def make_scheduler(scheduler_name: str, optimizer, **kwargs):
    name = scheduler_name.lower()

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get("T_max", 200))

    elif name == "onecycle":
        if "total_steps" not in kwargs:
            raise ValueError("OneCycleLR requires 'total_steps' in kwargs")
        return optim.lr_scheduler.OneCycleLR(
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
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
            threshold=kwargs.get("threshold", 1e-4),
            min_lr=kwargs.get("min_lr", 1e-7),
        )

    elif name in ("step", "multistep", "steplr", "multisteplr"):
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get("milestones", [60, 90]),
            gamma=kwargs.get("gamma", 0.1),
        )

    elif name in ("warmup_cosine", "warmupcosine"):
        warmup_epochs = kwargs.get("warmup_epochs", 5)
        T_max = kwargs.get("T_max", 100)
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, T_max - warmup_epochs),
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs],
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# =====================================================================
# データセットごとのプリセット (★ 2026-08-23 暫定値、要pilot後確定)
#
# CTRL_CONFIG とは異なり、これらは「原論文の設定を踏襲する」方針の対象。
# batch_size は原論文 Table 7 の値をそのまま使用。num_epochs / reach_threshold
# は原論文が明記していない (patience=16のみで停止) ため暫定値。pilot結果を見て
# 確定させること。
# =====================================================================

DATASET_CONFIGS = {
    "adult": dict(
        task_type="binclass",
        n_classes=1,
        num_epochs=100,          # ★暫定。要pilot確認 (burn-inは約20epochで完了する見込み)
        batch_size=256,          # 原論文 Table 7
        reach_threshold=0.84,    # ★暫定 (原論文のFT-Transformer accuracy=0.859を参考に少し下げた値。要pilot後確定)
        scheduler_presets={
            "cosine": dict(T_max=100),
            "onecycle": dict(max_lr=1e-3, pct_start=0.1, anneal_strategy="cos",
                              div_factor=25.0, final_div_factor=1e4),
            "plateau": dict(mode="min", factor=0.5, patience=8, threshold=1e-4, min_lr=1e-7),
            "step": dict(milestones=[60, 90], gamma=0.1),
            "warmup_cosine": dict(warmup_epochs=5, T_max=100),
        },
    ),
    "california_housing": dict(
        task_type="regression",
        n_classes=1,
        num_epochs=100,          # ★暫定。要pilot確認 (burn-inが約38epochと長め、要最優先チェック)
        batch_size=256,          # 原論文 Table 7
        reach_threshold=0.50,    # ★暫定 (原論文のFT-Transformer RMSE=0.459を参考にやや緩めた値。低いほど良い。要pilot後確定)
        scheduler_presets={
            "cosine": dict(T_max=100),
            "onecycle": dict(max_lr=1e-3, pct_start=0.1, anneal_strategy="cos",
                              div_factor=25.0, final_div_factor=1e4),
            "plateau": dict(mode="min", factor=0.5, patience=8, threshold=1e-4, min_lr=1e-7),
            "step": dict(milestones=[60, 90], gamma=0.1),
            "warmup_cosine": dict(warmup_epochs=5, T_max=100),
        },
    ),
    "covtype": dict(
        task_type="multiclass",
        n_classes=7,
        num_epochs=100,          # ★暫定。要pilot確認 (burn-inは約5.5epochで完了する見込み、画像実験と同程度)
        batch_size=1024,         # 原論文 Table 7
        reach_threshold=0.90,    # ★暫定 (原論文のFT-Transformer accuracy=0.970を参考にかなり下げた値。要pilot後確定)
        scheduler_presets={
            "cosine": dict(T_max=100),
            "onecycle": dict(max_lr=1e-3, pct_start=0.1, anneal_strategy="cos",
                              div_factor=25.0, final_div_factor=1e4),
            "plateau": dict(mode="min", factor=0.5, patience=8, threshold=1e-4, min_lr=1e-7),
            "step": dict(milestones=[60, 90], gamma=0.1),
            "warmup_cosine": dict(warmup_epochs=5, T_max=100),
        },
    ),
}


# =====================================================================
# 訓練 loop 1 run 分
# =====================================================================

def _compute_metric(task_type: str, logits_or_pred: torch.Tensor, y: torch.Tensor,
                     preprocessor: Optional[TabularPreprocessor] = None) -> float:
    """accuracy (binclass/multiclass) または RMSE (regression, 元スケールに逆変換) を返す。"""
    if task_type == "binclass":
        pred = (torch.sigmoid(logits_or_pred) >= 0.5).float()
        return (pred == y).float().mean().item()
    elif task_type == "multiclass":
        pred = logits_or_pred.argmax(dim=1)
        return (pred == y).float().mean().item()
    elif task_type == "regression":
        assert preprocessor is not None
        pred_orig = preprocessor.denormalize_y(logits_or_pred.detach().cpu())
        y_orig = preprocessor.denormalize_y(y.detach().cpu())
        return float(torch.sqrt(torch.mean((pred_orig - y_orig) ** 2)).item())
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def _is_better(task_type: str, candidate: float, best: float, min_delta: float) -> bool:
    higher_is_better = TASK_METRIC_HIGHER_IS_BETTER[task_type]
    if higher_is_better:
        return candidate > best + min_delta
    else:
        return candidate < best - min_delta


def train_one_seed_tabular(
    *,
    dataset: str,
    train_loader,
    val_loader,
    test_loader,
    meta: dict,
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
    log_every: int = 50,
    extra_config: Optional[dict] = None,
    controller_config: Optional[dict] = None,
    gala_config: Optional[dict] = None,
    cwgd_config: Optional[dict] = None,
    optimizer_override: Optional[dict] = None,
    save_dir: Optional[str] = None,
    method_label: Optional[str] = None,
) -> dict:
    """1 seed 分の訓練 (train_v5_multidataset_schedulers.py の train_one_seed を
    表データ・FT-Transformer用に書き換えた版。ロジックの骨格 (CTRL適用箇所・
    reach判定・test-once評価・W&Bロギング) は画像版と揃えてある。

    controller_config / gala_config / cwgd_config は互いに排他。
    ★ GALA は AdamW backbone (model.make_default_optimizer()) を使わず、専用の
    SGD(momentum) backbone に置き換わる (GALA_TABULAR_CONFIG のコメント参照。
    「同一 optimizer backbone 上での比較」ではなくなる点に注意)。CWGD は CTRL と
    同じく AdamW backbone にそのまま相乗りする。
    """
    set_seed(seed)
    task_type = meta["task_type"]
    higher_is_better = TASK_METRIC_HIGHER_IS_BETTER[task_type]

    use_gala = gala_config is not None
    use_cwgd = cwgd_config is not None
    use_ctrl = controller_config is not None
    assert sum([use_gala, use_cwgd, use_ctrl]) <= 1, (
        "controller_config / gala_config / cwgd_config は互いに排他です"
    )

    model = build_ft_transformer(meta).to(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    if task_type == "binclass":
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    gala_controller = None
    if use_gala:
        gala_controller = GALAController(**gala_config)
        optimizer = gala_controller.build_optimizer(model)
        scheduler = None
    else:
        if optimizer_override is not None:
            # ★CTRL v6診断実験専用: AdamW backboneの代わりにSGDを使う
            # (SGD_DIAG_TABULAR_CONFIG参照)。
            optimizer = torch.optim.SGD(model.parameters(), **optimizer_override)
        else:
            optimizer = model.make_default_optimizer()
        scheduler = make_scheduler(scheduler_name, optimizer, **scheduler_kwargs)

    controller = ContinuousStateKurtosisController(**controller_config) if use_ctrl else None

    cwgd_controller = None
    cwgd_per_sample_fn = None
    if use_cwgd:
        cwgd_cfg = dict(cwgd_config)
        if cwgd_cfg.get("refresh_interval_steps") is None:
            steps_per_epoch = len(train_loader)
            refresh_epochs = max(1, num_epochs // 8)
            cwgd_cfg["refresh_interval_steps"] = steps_per_epoch * refresh_epochs
        cwgd_controller = CWGDController(**cwgd_cfg)
        cwgd_per_sample_fn = make_ft_transformer_per_sample_loss_fn(criterion)

    name = scheduler_name.lower()
    is_onecycle = name == "onecycle"
    is_plateau = name in ("plateau", "reduce_on_plateau", "reducelronplateau")

    config = {
        "dataset": dataset,
        "seed": seed,
        "method": scheduler_name.upper(),
        "scheduler_name": scheduler_name,
        "scheduler_kwargs": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                             for k, v in scheduler_kwargs.items()},
        "task_type": task_type,
        "num_epochs": num_epochs,
        "optimizer": "AdamW (rtdl default)",
        "model": "FTTransformer (rtdl_revisiting_models, default kwargs)",
        "reach_threshold": reach_threshold,
        "patience": patience,
        "min_delta": min_delta,
        **({"ctrl_" + k: v for k, v in controller_config.items()} if controller_config else {}),
        **({"gala_" + k: v for k, v in gala_config.items()} if gala_config else {}),
        **({"cwgd_" + k: v for k, v in cwgd_cfg.items()} if use_cwgd else {}),
    }
    if extra_config:
        config.update(extra_config)

    run = wandb.init(project=project, name=run_name, config=config, reinit=True)

    reach_epoch = None
    reach_step = None
    final_metric = None

    best_val_metric = float("-inf") if higher_is_better else float("inf")
    best_model_state = None
    best_val_epoch = None
    no_improve_count = 0

    # ★ pilot診断用: CTRL適用時のkurtosis/ratio/current_multの推移を毎stepローカルにも
    # 保持しておく (W&Bを見に行かなくても、burn-in直後のover-damping有無をCSVで直接確認できる)。
    ctrl_trace = [] if use_ctrl else None

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (x_num, x_cat, y) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + batch_idx

            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            optimizer.zero_grad()
            out = forward_ft_transformer(model, x_num, x_cat)
            loss = criterion(out, y)
            loss.backward()

            ctrl_info = None
            cwgd_info = None
            gala_info = None
            if use_ctrl:
                gmag = collect_grad_magnitudes(model, clip_max=None)
                k_t = excess_kurtosis(gmag) if gmag.numel() > 0 else float("nan")
                ctrl_info = controller.update(k_t, optimizer, scheduler=scheduler)
                ctrl_trace.append({
                    "global_step": global_step, "epoch": epoch,
                    "k_t": ctrl_info.get("k_t"), "k_ewm": ctrl_info.get("k_ewm"),
                    "baseline": ctrl_info.get("baseline"), "ratio": ctrl_info.get("ratio"),
                    "current_mult": ctrl_info.get("current_mult"), "action": ctrl_info.get("action"),
                })
            elif use_cwgd:
                cwgd_info = cwgd_controller.update(
                    model, criterion, (x_num, x_cat), y,
                    optimizer=optimizer, scheduler=scheduler,
                    forward_fn=forward_ft_transformer,
                    per_sample_loss_fn=cwgd_per_sample_fn,
                )

            if use_gala:
                # GALA は追加逆伝播 → eta_n 算出 → optimizer.step() までを内部で
                # 一括して行う (gala_optimizer.py 参照)。以下の optimizer.step()
                # は呼ばない。
                gala_info = gala_controller.step(
                    model, (x_num, x_cat), y, criterion, optimizer,
                    forward_fn=forward_ft_transformer,
                )
            else:
                optimizer.step()

            if is_onecycle:
                scheduler.step()

            if batch_idx % log_every == 0:
                log_dict = {
                    "train/loss": loss.item(),
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                }
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

        model.eval()
        val_loss_sum, val_total = 0.0, 0
        all_out, all_y = [], []
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                out = forward_ft_transformer(model, x_num, x_cat)
                loss = criterion(out, y)
                val_loss_sum += loss.item() * x_num.size(0)
                val_total += x_num.size(0)
                all_out.append(out.detach().cpu())
                all_y.append(y.detach().cpu())

        val_loss = val_loss_sum / val_total
        val_metric = _compute_metric(
            task_type, torch.cat(all_out), torch.cat(all_y),
            preprocessor=meta["preprocessor"],
        )
        final_metric = val_metric

        reached = (val_metric >= reach_threshold) if higher_is_better else (val_metric <= reach_threshold)
        if reach_epoch is None and reached:
            reach_epoch = epoch
            reach_step = (epoch + 1) * len(train_loader) - 1

        if _is_better(task_type, val_metric, best_val_metric, min_delta):
            best_val_metric = val_metric
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        if scheduler is not None:
            if is_plateau:
                scheduler.step(val_loss)
            elif not is_onecycle:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_end_step = (epoch + 1) * len(train_loader) - 1
        metric_name = "val/rmse" if task_type == "regression" else "val/acc"

        wandb.log({
            "val/loss": val_loss, metric_name: val_metric,
            "lr": current_lr, "epoch": epoch,
        }, step=epoch_end_step)

        if no_improve_count >= patience:
            break

    stop_epoch = epoch

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    final_test_metric = None
    if test_loader is not None:
        model.eval()
        all_out, all_y = [], []
        with torch.no_grad():
            for x_num, x_cat, y in test_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                out = forward_ft_transformer(model, x_num, x_cat)
                all_out.append(out.detach().cpu())
                all_y.append(y.detach().cpu())
        final_test_metric = _compute_metric(
            task_type, torch.cat(all_out), torch.cat(all_y),
            preprocessor=meta["preprocessor"],
        )
        test_metric_name = "test/rmse_final" if task_type == "regression" else "test/acc_final"
        wandb.log({
            test_metric_name: final_test_metric,
            "train/stop_epoch": stop_epoch,
            "train/best_val_epoch": best_val_epoch,
            "train/best_val_metric": best_val_metric,
            "epoch": stop_epoch,
        }, step=epoch_end_step)

    # ★ pilot診断用: CTRLのkurtosis/ratio/current_multの推移をローカルCSVにも保存する
    # (W&Bを見に行かなくても、burn-in直後にover-dampingが起きていないかすぐ確認できる)。
    if use_ctrl and ctrl_trace and save_dir is not None:
        import pandas as pd
        os.makedirs(save_dir, exist_ok=True)
        label = method_label or scheduler_name  # CTRL実行時はscheduler_nameが"cosine"等の
        # 実体スケジューラ名に置き換わっているため、呼び出し元のmethod_label ("ctrl") を優先する
        trace_path = os.path.join(
            save_dir, f"{dataset}_{label}_seed{seed}_ctrl_trace.csv"
        )
        pd.DataFrame(ctrl_trace).to_csv(trace_path, index=False)
        print(f"  [CTRL trace] saved to: {trace_path} "
              f"(burnin_steps={controller_config['burnin_steps']}, "
              f"baseline={controller.baseline}, final_mult={controller.current_mult:.4f}, "
              f"interventions={controller.interventions})")

    # ★ pilot実行でのVRAM実測用。GPU2枚を他の人と融通しながら使う運用のため、
    # このrunがどれだけVRAMを使ったか毎回確認できるようにしておく。
    peak_mem_alloc_mb = None
    peak_mem_reserved_mb = None
    if str(device).startswith("cuda"):
        peak_mem_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_mem_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f"  [GPU] peak_memory_allocated={peak_mem_alloc_mb:.1f} MB, "
              f"peak_memory_reserved={peak_mem_reserved_mb:.1f} MB "
              f"(device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        wandb.log({
            "gpu/peak_memory_allocated_mb": peak_mem_alloc_mb,
            "gpu/peak_memory_reserved_mb": peak_mem_reserved_mb,
        }, step=epoch_end_step)

    run_id = run.id
    wandb.finish()

    return {
        "dataset": dataset,
        "seed": seed, "run_id": run_id,
        "method": scheduler_name.upper(),
        "scheduler_name": scheduler_name,
        "task_type": task_type,
        "reach": 1 if reach_epoch is not None else 0,
        "reach_threshold": reach_threshold,
        "reach_epoch": reach_epoch, "reach_step": reach_step,
        "final_val_metric": final_metric,
        "best_val_metric": best_val_metric,
        "best_val_epoch": best_val_epoch,
        "stop_epoch": stop_epoch,
        "final_test_metric": final_test_metric,
        "peak_memory_allocated_mb": peak_mem_alloc_mb,
        "peak_memory_reserved_mb": peak_mem_reserved_mb,
    }


# =====================================================================
# Multi-seed runner
# =====================================================================

def run_schedulers_multiseed_tabular(
    dataset: str,
    scheduler_names: List[str],
    *,
    protocol: str = "noes",
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    device: str = "cuda",
    project: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    min_delta: Optional[float] = None,
    data_root: str = DEFAULT_DATA_ROOT,
    data_source: str = "official",
    save_dir: Optional[str] = None,
    optimizer_override: Optional[dict] = None,
) -> list:
    """dataset × scheduler_names × seeds を実行する (本実験)。

    Args:
        dataset: "adult" / "california_housing" / "covtype"
        scheduler_names: ["ctrl", "cosine", "onecycle", "plateau", "step", "warmup_cosine"]
        protocol: "noes" (early stop 実質なし) / "es2" (patience=20, min_delta=0.001、
                  画像実験と同一条件で比較するためのプロジェクト独自定義。原論文の
                  patience=16とは異なる点に注意)
        num_epochs/batch_size: None なら DATASET_CONFIGS の既定値を使用
        data_source: "official" (原論文の公式split, 既定) / "sklearn_fallback"
                     (data.tar.gzが無い場合のパイプライン動作確認用、公式splitではない)
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. choices={list(DATASET_CONFIGS)}")
    cfg = DATASET_CONFIGS[dataset]

    num_epochs = num_epochs or cfg["num_epochs"]
    batch_size = batch_size or cfg["batch_size"]

    if protocol == "noes":
        patience = patience if patience is not None else (num_epochs + 1)
        min_delta = min_delta if min_delta is not None else 0.0
    elif protocol == "es2":
        # ★ 画像実験のES2定義 (patience=20, min_delta=0.001) をそのまま踏襲。
        # 原論文自身のpatience=16とは異なる (意図的、モダリティ横断の比較条件を揃えるため)。
        patience = patience if patience is not None else 20
        min_delta = min_delta if min_delta is not None else 0.001
    else:
        raise ValueError(f"Unknown protocol: {protocol} (choices: noes, es2)")

    project = project or (
        f"KurtosisEWMController-tabular-{dataset}-{protocol}-sgddiag"
        if optimizer_override is not None
        else f"KurtosisEWMController-tabular-{dataset}-{protocol}"
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, meta = make_tabular_loaders(
        dataset, batch_size=batch_size, data_root=data_root, data_source=data_source,
    )
    meta["n_classes"] = cfg["n_classes"]
    print(f"[{dataset}] train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
          f"test={len(test_loader.dataset)}, task_type={meta['task_type']}, "
          f"n_cont_features={meta['n_cont_features']}, cat_cardinalities={meta['cat_cardinalities']}, "
          f"protocol={protocol} (patience={patience}, min_delta={min_delta}), "
          f"data_source={data_source}")

    total_steps = num_epochs * len(train_loader)

    all_results = []
    for scheduler_name in scheduler_names:
        lname = scheduler_name.lower()
        is_ctrl = lname == "ctrl"
        is_gala = lname == "gala"
        is_cwgd = lname == "cwgd"

        if is_ctrl:
            effective_scheduler, effective_kwargs = "cosine", dict(T_max=num_epochs)
            ctrl_cfg, gala_cfg, cwgd_cfg_local = CTRL_CONFIG, None, None
        elif is_gala:
            # GALA は専用 SGD backbone を使う (GALA_TABULAR_CONFIG のコメント参照)。
            # scheduler は使わないので effective_kwargs は空でよい。
            effective_scheduler, effective_kwargs = "gala", {}
            ctrl_cfg, cwgd_cfg_local = None, None
            gala_cfg = dict(GALA_TABULAR_CONFIG)
        elif is_cwgd:
            effective_scheduler, effective_kwargs = "cosine", dict(T_max=num_epochs)
            ctrl_cfg, gala_cfg = None, None
            cwgd_cfg_local = CWGD_CONFIG
        else:
            effective_scheduler = scheduler_name
            effective_kwargs = cfg["scheduler_presets"][lname].copy()
            ctrl_cfg = gala_cfg = cwgd_cfg_local = None

        if effective_scheduler.lower() == "onecycle" and "total_steps" not in effective_kwargs:
            effective_kwargs["total_steps"] = total_steps

        # ★2026-08-24修正: cosine/warmup_cosineのT_maxはDATASET_CONFIGSのプリセット
        # (固定値、暫定num_epochs=100に合わせて書かれたもの)ではなく、常に実行時の
        # num_epochsで上書きする。CTRL側は元々 dict(T_max=num_epochs) で動的に
        # 設定されていたが、COSINE/WARMUP_COSINE側はプリセットのT_max=100が
        # 残ったままだったため、--epochsで上書きした場合(pilot実行など)に
        # CTRLとCOSINEでcosine周期が食い違い、不公平な比較になっていた
        # (onecycleのtotal_stepsは元々動的注入されていたのと同じ扱いにする)。
        if effective_scheduler.lower() in ("cosine", "warmup_cosine", "warmupcosine"):
            effective_kwargs["T_max"] = num_epochs

        # ★2026-08-24追加修正: T_maxと全く同じ理由でstep(MultiStepLR)のmilestonesも
        # 実行時のnum_epochsから動的に導出する。プリセットのmilestones=[60, 90]は
        # 暫定num_epochs=100(60%, 90%地点)に合わせて書かれた固定値で、--epochsで
        # 上書きした場合に追従しない。pilot(epochs=20/40/70)のようにnum_epochsが
        # 60未満だと一度もdecayが起きず、stepスケジューラが事実上定数LRになって
        # しまう(Adult/Covertypeで発覚)。60%/90%地点という原設定の比率を保ったまま
        # num_epochsに合わせてスケールする。
        if effective_scheduler.lower() in ("step", "multistep", "steplr", "multisteplr"):
            effective_kwargs["milestones"] = sorted(set(
                max(1, round(num_epochs * frac)) for frac in (0.6, 0.9)
            ))

        for seed in seeds:
            print(f"\n========== [{dataset}] seed={seed}: {scheduler_name.upper()} "
                  f"(optimizer_override={optimizer_override}) ==========")
            r = train_one_seed_tabular(
                dataset=dataset,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                meta=meta,
                device=device, seed=seed,
                run_name=(
                    f"{dataset}-fttransformer-{protocol}-{scheduler_name}-seed{seed}"
                    + ("-sgdopt" if optimizer_override is not None else "")
                ),
                project=project,
                scheduler_name=effective_scheduler,
                scheduler_kwargs=effective_kwargs,
                num_epochs=num_epochs,
                reach_threshold=cfg["reach_threshold"],
                patience=patience,
                min_delta=min_delta,
                extra_config={"protocol": protocol, "batch_size": batch_size,
                              "data_source": data_source,
                              "optimizer_override": optimizer_override},
                controller_config=ctrl_cfg,
                gala_config=gala_cfg,
                cwgd_config=cwgd_cfg_local,
                optimizer_override=optimizer_override,
                save_dir=save_dir,
                method_label=scheduler_name,
            )
            if is_ctrl:
                r["method"] = "CTRL"
            elif is_gala:
                r["method"] = "GALA"
            elif is_cwgd:
                r["method"] = "CWGD"
            if optimizer_override is not None:
                r["method"] = f"{r['method']}-SGD"
            all_results.append(r)
            test_str = (f", final_test_metric={r['final_test_metric']:.4f}"
                        if r.get("final_test_metric") is not None else "")
            print(f"  -> reach={r['reach']} reach_epoch={r['reach_epoch']} "
                  f"final_val_metric={r['final_val_metric']:.4f}{test_str}")

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
        description="Tabular (FT-Transformer) scheduler comparison (CTRL v5 generality check, Axis B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(TABULAR_DATASETS))
    parser.add_argument("--protocol", type=str, default="noes", choices=["noes", "es2"])
    parser.add_argument(
        "--schedulers", type=str, nargs="+",
        default=["ctrl", "cosine"],
        choices=["ctrl", "cosine", "onecycle", "plateau", "step", "warmup_cosine",
                 "gala", "cwgd"],
        help="★このv6診断フォークでは既定を[ctrl, cosine]に変更済み(元スクリプトは "
             "8手法全部が既定)。gala は AdamW backbone ではなく専用 SGD backbone を "
             "使う(GALA_TABULAR_CONFIG 参照、--optimizer_override とは無関係)。"
             "cwgd は必ず先に --epochs を小さくした単一 seed パイロットでCOSINEに "
             "対して悪化しないか確認すること。★--optimizer_override sgd と "
             "gala/cwgd の併用は想定外なので避けること(gala は無視される、cwgd は "
             "意図せずSGD backboneに変わってしまう)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--cuda_visible_devices", type=str, default=None,
        help="例: '0' か '1'。指定した場合、シェル側のCUDA_VISIBLE_DEVICES設定より"
             "優先してこの値を使う (2枚あるGPUを実行のたびに切り替えたい場合用)。"
             "省略時はシェルの環境変数、それも無ければDEFAULT_CUDA_VISIBLE_DEVICES"
             f"('{DEFAULT_CUDA_VISIBLE_DEVICES}') を使用。",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="None なら DATASET_CONFIGS の既定 epoch 数を使用")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="None なら --protocol に応じた既定値を使用")
    parser.add_argument("--min_delta", type=float, default=None,
                        help="None なら --protocol に応じた既定値を使用")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                        help="data.tar.gz を展開したディレクトリ")
    parser.add_argument("--data_source", type=str, default="official",
                        choices=["official", "sklearn_fallback"],
                        help="official: 原論文の公式split (本実験用)。sklearn_fallback: "
                             "data.tar.gzが無い場合のパイプライン動作確認用 (公式splitではない)")
    parser.add_argument(
        "--optimizer_override", type=str, default="none", choices=["none", "sgd"],
        help="★CTRL v6診断実験専用オプション。'sgd'を指定すると、AdamW backbone "
             "(model.make_default_optimizer()) の代わりに SGD_DIAG_TABULAR_CONFIG "
             "(lr=0.1, momentum=0.9, nesterov=True, wd=0、GALA_TABULAR_CONFIGと同一) "
             "を使う。CTRL自体のハイパラ(CTRL_CONFIG)は変更しない。'none'(既定)なら "
             "従来通りAdamWのまま。--schedulers には ctrl/cosine 等のみ指定すること "
             "(gala/cwgdとの併用は非対応、--schedulers のhelp参照)。",
    )
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    # --cuda_visible_devices が指定されていれば、他の何よりも優先してここで上書きする。
    # torch.cuda.* はまだ一度も呼ばれていないので (このファイルはtorchをimportして
    # いるだけでCUDAコンテキストはまだ初期化されていない)、ここで設定すれば確実に反映される。
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    print(f"dataset={args.dataset}, protocol={args.protocol}, schedulers={args.schedulers}, "
          f"seeds={args.seeds}, device={args.device}, epochs={args.epochs}, "
          f"data_source={args.data_source}, optimizer_override={args.optimizer_override}")

    results = run_schedulers_multiseed_tabular(
        dataset=args.dataset,
        scheduler_names=args.schedulers,
        protocol=args.protocol,
        seeds=tuple(args.seeds),
        device=args.device,
        project=args.project,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        data_root=args.data_root,
        data_source=args.data_source,
        save_dir=args.save_dir,
        optimizer_override=(
            dict(SGD_DIAG_TABULAR_CONFIG) if args.optimizer_override == "sgd" else None
        ),
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
        test_metric_mean=("final_test_metric", "mean"),
        test_metric_std=("final_test_metric", "std"),
    )
    print(summary.to_string())

    if args.save_dir is not None:
        out_path = os.path.join(args.save_dir, f"{args.dataset}_{args.protocol}_summary.csv")
        summary.to_csv(out_path)
        print(f"\nSaved summary to: {out_path}")
