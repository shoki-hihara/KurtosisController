"""
gala_optimizer.py — SGD-GALA (Gradient Alignment-based Learning rate Adaptation)

出典: Jiang, Kavis, Mokhtari, "Online Learning-guided Learning Rate Adaptation
      via Gradient Alignment" (arXiv:2506.08419, 2025), 論文中の式(11) の
      簡略版 (SGD-GALA) を実装する。

CTRL (kurtosis-based LR controller, ctrl_v5_continuous.py) との比較実験用の
ベースライン手法その1 (優先度1)。CTRL とは異なり、GALA は既存の
torch.optim.lr_scheduler を一切使わず、**毎 step 自律的に学習率を決定する**。
そのため統合方法も CTRL (「既存 scheduler の base_lrs を書き換える」) とは異なり、
「optimizer.param_groups の lr を直接書き換えてから optimizer.step() を呼ぶ」
という形になる。

-----------------------------------------------------------------------
アルゴリズム (引継ぎタスク仕様書のとおり)
-----------------------------------------------------------------------
通常の SGD 更新: x_{t+1} = x_t - eta_t * g_t(x_t)

同一ミニバッチ xi_{t+1} を使い、更新後の x_{t+1} と更新前の x_t の両方で
勾配を評価する:
    g'_{t+1} = grad f(x_{t+1}; xi_{t+1})   … 通常の (次の) 勾配計算そのもの
    g'_t     = grad f(x_t;     xi_{t+1})   … ★追加の逆伝播 (旧パラメータ + 新バッチ)

ローカル Lipschitz 推定:
    L_t = || g'_{t+1} - g'_t || / || x_{t+1} - x_t ||

学習率更新 (FTRL 型、累積和):
    eta_{t+1} = sum_{s=0}^{t} <g'_{s+1}, g'_s>  /  sum_{s=0}^{t} L_s * || g_s(x_s) ||^2

ここで g_s(x_s) は「実際にステップ s→s+1 の更新に使われた、その時点の
ミニバッチによる勾配」(g'_{s+1}/g'_s とは別の、通常の SGD 勾配) である。

-----------------------------------------------------------------------
本実装での index の取り方 (docstring 中で t, s は上式のまま、
コード変数は反復回数 n で表す)
-----------------------------------------------------------------------
反復 n (params θ_n → θ_{n+1} への更新) ごとに:
  1. 現在のミニバッチ B_n で forward+backward → g_n = grad f(θ_n; B_n)。
     これは上式の g'_{n} (= g'_{t+1}, t=n-1) であると同時に、
     「実際の更新に使われる勾配」g_n(θ_n) でもある (この2つは同一の量)。
  2. 直前の反復で保存しておいた「1つ前のパラメータ」θ_{n-1} を一時的に
     モデルへロードし、**同じバッチ B_n** で forward+backward
     → g_prev = grad f(θ_{n-1}; B_n)。これが g'_{n-1} (= g'_t, t=n-1)。
     このステップだけ、追加の逆伝播が必要になる (計算コスト約2倍)。
  3. L_{n-1} = || g_n - g_prev || / || θ_n - θ_{n-1} ||
  4. 累積和を更新:
       numerator   += <g_n, g_prev>
       denominator += L_{n-1} * || g_{n-1}(θ_{n-1}) ||^2
     ただし g_{n-1}(θ_{n-1}) は「1つ前の反復で実際に更新に使われた勾配」
     であり、これは1つ前の反復で計算した g_{n-1} (このモジュール内では
     self._prev_own_grad として保存してある) と同一の量。
  5. eta_n = numerator / denominator (反復0では history がないので eta_0 を使用)
  6. optimizer の lr を eta_n に設定し、g_n を使って実際のパラメータ更新
     θ_n → θ_{n+1} を行う (momentum/nesterov/weight_decay は通常の
     torch.optim.SGD に委譲する)。
  7. 次の反復のために θ_n (更新前のパラメータ) と g_n を保存する。

-----------------------------------------------------------------------
実装上の注意点 (論文に明記がないため、本実装で下した設計判断)
-----------------------------------------------------------------------
* **BatchNorm**: ステップ2の「追加の逆伝播」は同一バッチを旧パラメータで
  再度 forward するため、model.train() のままだと BN の running statistics
  が同一バッチで二重に更新されてしまう (かつ古い conv 重みに対する統計という
  意味のない値で汚染される)。これを避けるため、追加の逆伝播の間だけ
  model.eval() にして running stats を使わせる (更新もしない)。
  Dropout も同様に eval() 中は無効化される (これは望ましい副作用: 追加の
  逆伝播でも同じ dropout mask を保証する必要はなく、むしろ確率的要素を
  減らした方が L_t 推定が安定する)。

* **fair comparison (wd / nesterov)**: 引継ぎ仕様書の指示どおり、GALA の
  更新式自体に momentum/weight_decay に関する理論的制約はないため、
  CTRL や他スケジューラ (COSINE/STEP/...) と同じ optimizer 設定
  (momentum=0.9, nesterov=True, weight_decay=5e-4) に相乗りさせる。
  これにより「同じ optimizer backbone 上で LR 制御機構だけを変える」という
  他ベースラインとの比較軸が揃う。GALA 原論文設定 (momentum=0.9, wd=0)
  での実行は `weight_decay=0.0` を渡すことで別途可能 (アブレーションとして
  実行する場合に使用)。

* **初期学習率 eta_0 のみ外部指定**: 論文の実験どおり、clipping は行わない。
  eta_0 は他ベースラインと揃えて base_lr=0.1 をデフォルトにしている
  (CTRL_FS_CONFIG 等と同じ「ハイパラ非再チューニング」方針に合わせるため)。

* **計算コスト**: 1 step あたり forward+backward が実質2回 (通常の1回 +
  追加の1回) になる。実験ログには `gala/extra_backward` 相当の情報を
  残さないが、呼び出し側 (train_one_seed 系) で "GALA は1 step 2 backward"
  である旨をログ・報告書に明記すること。

* **数値安定性**: 分母 (denominator) が 0 に極めて近い場合 (学習初期、
  L_t がほぼ0 になるケース) はゼロ除算を避けるため epsilon を加える。
  論文はこの点に言及していないが、実装上必須の安全策。
"""

from __future__ import annotations

import copy
from typing import Optional

import torch


class GALAController:
    """SGD-GALA の学習率適応コントローラ。

    使い方 (train loop 内):
        gala = GALAController(eta0=0.1, momentum=0.9, nesterov=True,
                               weight_decay=5e-4)
        optimizer = gala.build_optimizer(model)   # 内部で SGD を生成

        for imgs, labels in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()                      # g_n がここで計算される

            info = gala.step(model, imgs, labels, criterion, optimizer)
            # ↑ 内部で: 追加の逆伝播 → eta_n 計算 → optimizer.param_groups の
            #    lr を書き換え → optimizer.step() まで実行する
            #    (呼び出し側は optimizer.step() を別途呼ばなくてよい)
    """

    def __init__(
        self,
        eta0: float = 0.1,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 5e-4,
        eps: float = 1e-12,
        eta_min: Optional[float] = None,
        eta_max: Optional[float] = None,
        grad_clip: Optional[float] = None,
    ):
        """
        Args:
            eta0: 初期学習率 (history がない最初の1 step のみ使用)。
            momentum, nesterov, weight_decay: build_optimizer() で使う
                torch.optim.SGD のハイパラ。GALA の更新式自体には制約がない
                ため、比較対象 (CTRL 等) の optimizer 設定に揃えるのが既定。
            eps: 分母のゼロ除算防止用の小さい定数。
            eta_min, eta_max: 論文は clipping を行わないが、実運用上の
                発散防止として任意で下限/上限を指定できる (None ならクリップ
                しない = 論文どおり)。デフォルトは None (原論文どおり無効)。
            grad_clip: None 以外なら、追加の逆伝播で得た g_prev に対して
                `torch.nn.utils.clip_grad_norm_` と同じ global-norm clipping
                を内部で適用する (呼び出し側で g_n に対して同じ max_norm の
                クリッピングを既に行っている場合に指定する。例: text/LSTM
                軸の Zaremba et al. 2014 レシピは grad clip=5.0 が前提)。
                これを指定しないと、L_t/alignment の計算に使う g_n は
                クリップ済み・g_prev は未クリップという不整合な組み合わせに
                なってしまう (呼び出し側は既に g_n を clip してから
                step() を呼ぶ設計のため)。デフォルト None ( CIFAR-100 等の
                画像軸のようにそもそも勾配クリッピングを行わないレシピでは
                指定不要)。
        """
        self.eta0 = eta0
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.eps = eps
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.grad_clip = grad_clip

        self._numerator = 0.0
        self._denominator = 0.0
        self._prev_params: Optional[list] = None      # theta_{n-1} (flat snapshot list)
        self._prev_own_grad: Optional[list] = None     # g_{n-1}(theta_{n-1})
        self.step_count = 0
        self.last_eta = eta0
        self.last_L = None
        self.last_alignment = None

    # -----------------------------------------------------------------
    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.SGD:
        return torch.optim.SGD(
            model.parameters(),
            lr=self.eta0,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay,
        )

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _snapshot_params(self, model: torch.nn.Module) -> list:
        return [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def _snapshot_grads(self, model: torch.nn.Module) -> list:
        return [
            (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for p in model.parameters()
        ]

    @torch.no_grad()
    def _load_params(self, model: torch.nn.Module, params: list) -> None:
        for p, saved in zip(model.parameters(), params):
            p.data.copy_(saved)

    @torch.no_grad()
    def _restore_grads(self, model: torch.nn.Module, grads: list) -> None:
        for p, g in zip(model.parameters(), grads):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

    @torch.no_grad()
    def _flat_diff_norm(self, a: list, b: list) -> float:
        sq = 0.0
        for x, y in zip(a, b):
            sq += float(((x - y) ** 2).sum().item())
        return sq ** 0.5

    @torch.no_grad()
    def _flat_norm(self, a: list) -> float:
        sq = 0.0
        for x in a:
            sq += float((x ** 2).sum().item())
        return sq ** 0.5

    @torch.no_grad()
    def _flat_inner(self, a: list, b: list) -> float:
        s = 0.0
        for x, y in zip(a, b):
            s += float((x * y).sum().item())
        return s

    # -----------------------------------------------------------------
    @staticmethod
    def _as_tuple(inputs):
        return inputs if isinstance(inputs, (tuple, list)) else (inputs,)

    def step(
        self,
        model: torch.nn.Module,
        inputs,
        labels: torch.Tensor,
        criterion,
        optimizer: torch.optim.Optimizer,
        forward_fn=None,
    ) -> dict:
        """1 step 分の GALA 更新を実行する。

        Args:
            inputs: 通常は画像テンソルなど単一の Tensor。FT-Transformer の
                (x_num, x_cat) のように model の forward が複数引数を取る
                場合は tuple/list で渡す (内部で model(*inputs) 相当として
                展開される)。
            forward_fn: None なら `model(*inputs)` (inputs が単一 Tensor
                なら `model(inputs)`) をそのまま使う。FT-Transformer の
                `forward_ft_transformer(model, x_num, x_cat)` のように、
                出力の squeeze や引数の前処理 (x_cat が空なら None を渡す等)
                が必要なモデルでは、`forward_fn(model, *inputs) -> outputs`
                を渡すこと。

        呼び出し前提: `loss.backward()` が直前に呼ばれ、`model.parameters()`
        の `.grad` に g_n = grad f(theta_n; B_n) が入っていること。

        この関数の内部で:
          - (history があれば) 追加の逆伝播で g_prev を計算し、eta_n を算出
          - optimizer.param_groups の lr を eta_n に設定
          - g_n を .grad に復元したうえで optimizer.step() を呼び、
            theta_n -> theta_{n+1} の実際の更新を行う
          - 次回のために theta_n, g_n を保存する

        Returns:
            dict: {"eta": eta_n, "L_t": ..., "alignment": ...,
                   "numerator": ..., "denominator": ...}
                  history がまだない最初の step では L_t/alignment は None。
        """
        self.step_count += 1
        was_training = model.training
        inputs_t = self._as_tuple(inputs)
        _forward = forward_fn if forward_fn is not None else (lambda m, *args: m(*args))

        # g_n = 現在の勾配 (呼び出し元の loss.backward() 直後)
        g_n = self._snapshot_grads(model)
        theta_n = self._snapshot_params(model)  # = x_{n} (更新前パラメータ)

        L_t = None
        alignment = None

        if self._prev_params is not None:
            # ----- 追加の逆伝播: g_prev = grad f(theta_{n-1}; B_n) -----
            # BN の running stats 汚染 / dropout の余計な確率性を避けるため
            # eval() で forward する (本モジュール独自の設計判断、docstring 参照)
            model.eval()
            self._load_params(model, self._prev_params)
            optimizer.zero_grad(set_to_none=False)
            outputs = _forward(model, *inputs_t)
            loss = criterion(outputs, labels)
            loss.backward()
            if self.grad_clip is not None:
                # 呼び出し側が g_n に適用済みのクリッピングと揃える
                # (docstring 参照: g_n/g_prev の整合性のため)。
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            g_prev = self._snapshot_grads(model)

            # ----- 復元: パラメータを theta_n に戻し、grad も g_n に戻す -----
            self._load_params(model, theta_n)
            self._restore_grads(model, g_n)
            if was_training:
                model.train()

            # ----- L_t, alignment, 累積和更新 -----
            denom_norm = self._flat_diff_norm(theta_n, self._prev_params)
            L_t = self._flat_diff_norm(g_n, g_prev) / (denom_norm + self.eps)
            alignment = self._flat_inner(g_n, g_prev)

            prev_own_grad_norm_sq = self._flat_norm(self._prev_own_grad) ** 2

            self._numerator += alignment
            self._denominator += L_t * prev_own_grad_norm_sq

            eta_n = self._numerator / (self._denominator + self.eps)
        else:
            eta_n = self.eta0

        if self.eta_min is not None:
            eta_n = max(eta_n, self.eta_min)
        if self.eta_max is not None:
            eta_n = min(eta_n, self.eta_max)

        # ----- 実際の更新: lr=eta_n, grad=g_n (momentum/nesterov/wd は SGD に委譲) -----
        for pg in optimizer.param_groups:
            pg["lr"] = eta_n
        optimizer.step()

        # ----- 次回のために保存 -----
        self._prev_params = theta_n
        self._prev_own_grad = g_n

        self.last_eta = eta_n
        self.last_L = L_t
        self.last_alignment = alignment

        return {
            "eta": eta_n,
            "L_t": L_t,
            "alignment": alignment,
            "numerator": self._numerator,
            "denominator": self._denominator,
        }
