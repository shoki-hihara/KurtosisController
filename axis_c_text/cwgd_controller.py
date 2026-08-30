"""
cwgd_controller.py — CWGD-Cosine (Curvature-Weighted Gradient Diversity)

出典: Hamza, Goel, "Curvature-Weighted Gradient Diversity: A Noise Measure
      for Geometry-Adaptive SGD Schedules" (arXiv:2606.30455, 2026), Algorithm 1。

CTRL (kurtosis-based LR controller) との比較実験用ベースライン手法その2
(優先度2)。CWGD は CTRL と同じく「既存の CosineAnnealingLR を base として、
その出力を毎 step 連続的に変調する」設計なので、統合方法は
ctrl_v5_continuous.ContinuousStateKurtosisController と同じパターンに
揃えてある (scheduler.base_lrs を書き換える)。

-----------------------------------------------------------------------
アルゴリズム (引継ぎタスク仕様書 Algorithm 1 のとおり)
-----------------------------------------------------------------------
1. 初期化時 (および Δ step ごとの再計算時) に Hutchinson probe
   (Rademacher ベクトル v、probe 数 P≈20) で Hessian 対角成分を推定する:
       Hv           ≈ (grad f(x + delta*v) - grad f(x)) / delta   … 有限差分
       lambda_hat_k ≈ mean_p [ v_p ⊙ Hv_p ]_k                     … Hutchinson 対角推定

2. 各 step で、ミニバッチ内の per-sample 勾配 g_i (i=1..B) を計算し、
       CWGD_t = 2 * sum_k ( sigma_hat_k^2 / lambda_hat_k )
   sigma_hat_k^2 はミニバッチ内サンプル勾配の k 次元方向の分散
   (1 ミニバッチ分の per-sample gradient のみで計算可能、1 epoch 通す必要なし)。

3. コサインスケジュールを変調:
       eta_t = eta_cos(t) / (1 + alpha * CWGD_t / CWGD_0),   alpha=1 (推奨)

4. Hutchinson probe は計算コスト削減のため、毎 step ではなく数 epoch ごと
   (論文では Delta ≈ T/8) に再計算する (staleness とのトレードオフ)。

-----------------------------------------------------------------------
★★★ 重要な留意事項 (引継ぎタスク仕様書より、実装・運用で必ず踏まえること) ★★★
-----------------------------------------------------------------------
原論文の実験は強凸な合成二次関数 (d=50) のみで行われており、著者自身が
「非凸の実ニューラルネットでは Hessian 推定の staleness (古さ) により
2〜14% 性能が悪化する」と明記している。CIFAR-100/ResNet18 にそのまま
適用しても効果が出ない可能性が高い。

→ 本実装をいきなり複数 seed・複数データセットのフル実験で使わず、まず
  単一 seed・小規模 (少 epoch) パイロット実験を行い、CTRL の既存ベース
  ライン (COSINE) に対して悪化しないかを確認してから本実験に進むこと。
  (呼び出し側の run_pilot_scheduler 等で `--epochs` を小さくして実行する)

-----------------------------------------------------------------------
実装上の注意点 (論文 Algorithm 1 に明記がないため、本実装で下した設計判断)
-----------------------------------------------------------------------
* **per-sample 勾配の計算方法**: `torch.func.vmap(torch.func.grad(...))` を
  使用する (2020年代の標準的な per-sample gradient 計算手法)。ResNet18
  クラスの中規模モデルではメモリ使用量が大きくなりうる
  (per-sample grad テンソルは概念上 batch_size × #params 分のメモリを要する:
  例えば batch=128, #params=11M, fp32 なら約 5.6GB)。**この計算コスト・
  メモリコストこそが「まずパイロットで確認せよ」という指示の主要因の一つ**
  であり、OOM する場合は `subsample_size` 引数でミニバッチの一部のみを
  使って分散を推定するフォールバックを用意してある (デフォルトは None =
  ミニバッチ全体を使う、論文に忠実な設定)。

* **BatchNorm**: per-sample 勾配計算は `functional_call` + `vmap` で
  batch_size=1 の forward を疑似的に行うため、BN 層が train mode のまま
  だと「batch_size=1 の分散」という無意味な統計になってしまう。これを
  避けるため、per-sample 勾配計算の間だけ model.eval() にし、BN の
  running statistics を固定値として使う (gala_optimizer.py の追加逆伝播
  と同じ設計判断)。

* **Hessian 対角成分の符号**: 原論文は強凸関数 (Hessian 対角 ≥ 0 が保証)
  のみを扱っているため、Hutchinson 推定値の符号については論じていない。
  非凸な実ニューラルネットでは鞍点等で対角成分が負になりうる。本実装では
  `lambda_hat_k` を `max(abs(lambda_hat_k), lambda_floor)` として扱い、
  「曲率の大きさ」として解釈する (符号反転や発散を防ぐための実装上の
  安全策。原論文には明記がないため、本実装独自の設計判断であることを
  論文・報告書に明記すること)。

* **sigma_hat_k^2 の定義**: ミニバッチ平均からの母分散 (biased variance,
  unbiased=False) を用いる。McCandlish et al. の gradient noise scale と
  同系統の標準的な定義 ([[project_ctrl_related_work]] 参照)。

* **CWGD_0 の定義**: 最初に (Hutchinson 初期化後) CWGD_t を計算した時点の
  値を CWGD_0 として固定する (原論文の "at initialization" を "controller
  が初めて有効な値を計算できた時点" と解釈)。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.func import functional_call, vmap, grad as func_grad


class CWGDController:
    """CWGD-Cosine の学習率変調コントローラ (CTRL と同じ「base scheduler の
    base_lrs を書き換える」パターンで使う)。

    使い方 (train loop 内、ctrl_v5_continuous の ContinuousStateKurtosisController
    と同じ呼び出しパターン):

        cwgd = CWGDController(alpha=1.0, num_probes=20,
                               refresh_interval_steps=steps_per_epoch * (num_epochs // 8))
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        for imgs, labels in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()

            info = cwgd.update(model, criterion, imgs, labels,
                                optimizer=optimizer, scheduler=scheduler)
            optimizer.step()
            # (epoch 末で scheduler.step() を呼ぶのは他手法と同じ)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        num_probes: int = 20,
        hutchinson_delta: float = 1e-3,
        refresh_interval_steps: int = 500,
        lambda_floor: float = 1e-6,
        subsample_size: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            alpha: eta_t = eta_cos(t) / (1 + alpha * CWGD_t/CWGD_0) の係数
                (論文推奨値 1.0)。
            num_probes: Hutchinson probe 数 P (論文目安 20)。
            hutchinson_delta: 有限差分のステップ幅 delta。
            refresh_interval_steps: Hutchinson 再計算の間隔 (step 数)。
                論文の Delta≈T/8 (epoch 単位) を呼び出し側で
                steps_per_epoch * (num_epochs // 8) 等に変換して渡す。
            lambda_floor: Hessian 対角推定値の絶対値の下限 (ゼロ除算・
                符号反転防止)。
            subsample_size: per-sample 勾配計算に使うサンプル数の上限。
                None ならミニバッチ全体 (論文に忠実)。OOM 時のフォール
                バック用 (パイロットで計算コストを確認した上で調整すること)。
            seed: Rademacher probe 生成用の乱数 seed (再現性確保用、任意)。
        """
        self.alpha = alpha
        self.num_probes = num_probes
        self.hutchinson_delta = hutchinson_delta
        self.refresh_interval_steps = refresh_interval_steps
        self.lambda_floor = lambda_floor
        self.subsample_size = subsample_size

        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

        self.lambda_diag: Optional[list] = None   # list[Tensor], model.parameters() と同じ shape
        self.cwgd0: Optional[float] = None
        self.step_count = 0

        self.original_base_lrs: Optional[list] = None
        self.current_mult = 1.0

        # 診断用ログ
        self.last_cwgd_t = None
        self.last_mult = 1.0
        self.last_hutchinson_refresh_step = None

    # -----------------------------------------------------------------
    # Hutchinson diagonal Hessian estimate
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _snapshot_params(self, model: torch.nn.Module) -> list:
        return [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def _load_params(self, model: torch.nn.Module, params: list) -> None:
        for p, saved in zip(model.parameters(), params):
            p.data.copy_(saved)

    def _rademacher_like(self, params: list) -> list:
        out = []
        for p in params:
            r = torch.randint(0, 2, p.shape, generator=self._gen, device="cpu").to(p.device)
            out.append((r.float() * 2 - 1))
        return out

    @staticmethod
    def _as_tuple(inputs):
        return inputs if isinstance(inputs, (tuple, list)) else (inputs,)

    def refresh_hutchinson(
        self,
        model: torch.nn.Module,
        criterion,
        inputs,
        labels: torch.Tensor,
        forward_fn=None,
    ) -> None:
        """calibration batch (現在のミニバッチ) を使って Hessian 対角成分を再推定する。

        num_probes+1 回の追加 forward+backward を要する (呼び出し側で
        refresh_interval_steps ごとにのみ呼ぶことでコストを償却する)。

        Args:
            inputs: 単一 Tensor、または (x_num, x_cat) のような tuple/list。
            forward_fn: None なら `model(*inputs)` をそのまま使う。
                FT-Transformer 等、出力の後処理や引数の前処理が必要な
                モデルでは `forward_fn(model, *inputs) -> outputs` を渡す
                (gala_optimizer.GALAController.step の forward_fn と同じ規約)。
        """
        was_training = model.training
        theta0 = self._snapshot_params(model)
        inputs_t = self._as_tuple(inputs)
        _forward = forward_fn if forward_fn is not None else (lambda m, *args: m(*args))

        # base gradient: grad f(x; batch)
        model.eval()  # BN running stats を汚染しないよう固定 (docstring 参照)
        model.zero_grad(set_to_none=False)
        outputs = _forward(model, *inputs_t)
        loss = criterion(outputs, labels)
        loss.backward()
        grad0 = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                 for p in model.parameters()]

        lambda_sum = [torch.zeros_like(p) for p in theta0]

        for _ in range(self.num_probes):
            v = self._rademacher_like(theta0)

            # x + delta*v
            for p, base, vi in zip(model.parameters(), theta0, v):
                p.data.copy_(base + self.hutchinson_delta * vi)
            model.zero_grad(set_to_none=False)
            outputs = _forward(model, *inputs_t)
            loss = criterion(outputs, labels)
            loss.backward()
            grad_pert = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                         for p in model.parameters()]

            for acc, gp, g0, vi in zip(lambda_sum, grad_pert, grad0, v):
                hv_approx = (gp - g0) / self.hutchinson_delta
                acc.add_(vi * hv_approx)

        # 復元
        self._load_params(model, theta0)
        model.zero_grad(set_to_none=False)
        if was_training:
            model.train()

        self.lambda_diag = [
            (acc / self.num_probes).abs_().clamp_min_(self.lambda_floor)
            for acc in lambda_sum
        ]
        self.last_hutchinson_refresh_step = self.step_count

    # -----------------------------------------------------------------
    # Per-sample gradient variance (CWGD_t)
    # -----------------------------------------------------------------
    def _compute_cwgd_t(
        self,
        model: torch.nn.Module,
        criterion,
        inputs,
        labels: torch.Tensor,
        per_sample_loss_fn=None,
    ) -> float:
        """per-sample 勾配分散から CWGD_t を計算する。

        Args:
            inputs: 単一 Tensor、または (x_num, x_cat) のような tuple/list。
            per_sample_loss_fn: None なら
                `functional_call(model, (params,buffers), sample_args)` を
                そのまま使い `criterion(pred, target)` を返す既定実装を使う
                (単一 Tensor 入力・後処理不要なモデル向け、CIFAR/MNIST 等)。
                FT-Transformer のように出力の squeeze や引数の前処理
                (x_cat が空なら None を渡す等) が必要なモデルでは
                `per_sample_loss_fn(model, params, buffers, *sample_args, target) -> scalar loss`
                を渡すこと (sample_args は batch 次元 1 の per-sample テンソル)。
        """
        was_training = model.training
        model.eval()  # per-sample forward (batch_size=1 相当) の BN 対策
        inputs_t = self._as_tuple(inputs)
        n = inputs_t[0].shape[0]

        if self.subsample_size is not None and n > self.subsample_size:
            idx = torch.randperm(n, generator=self._gen)[: self.subsample_size]
            idx = idx.to(inputs_t[0].device)
            sub_inputs_t = tuple(x[idx] for x in inputs_t)
            sub_labels = labels[idx]
        else:
            sub_inputs_t = inputs_t
            sub_labels = labels

        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        param_names = list(params.keys())

        if per_sample_loss_fn is not None:
            def compute_loss(params_dict, *sample_args_and_target):
                *sample_args, target = sample_args_and_target
                return per_sample_loss_fn(model, params_dict, buffers, *sample_args, target)
        else:
            def compute_loss(params_dict, *sample_args_and_target):
                *sample_args, target = sample_args_and_target
                batched_args = tuple(a.unsqueeze(0) for a in sample_args)
                tgt = target.unsqueeze(0)
                pred = functional_call(model, (params_dict, buffers), batched_args)
                return criterion(pred, tgt)

        in_dims = (None,) + tuple(0 for _ in sub_inputs_t) + (0,)
        per_sample_grad_fn = vmap(func_grad(compute_loss), in_dims=in_dims)
        per_sample_grads = per_sample_grad_fn(params, *sub_inputs_t, sub_labels)
        # per_sample_grads: dict[name] -> Tensor of shape (B_sub, *param.shape)

        cwgd_sum = 0.0
        for name, lam in zip(param_names, self.lambda_diag):
            g = per_sample_grads[name]
            var_k = torch.var(g, dim=0, unbiased=False)  # shape == param.shape
            cwgd_sum += float((var_k / lam).sum().item())

        if was_training:
            model.train()

        return 2.0 * cwgd_sum

    # -----------------------------------------------------------------
    # Public update() — CTRL 互換インタフェース
    # -----------------------------------------------------------------
    def update(
        self,
        model: torch.nn.Module,
        criterion,
        inputs,
        labels: torch.Tensor,
        optimizer,
        scheduler=None,
        forward_fn=None,
        per_sample_loss_fn=None,
    ) -> dict:
        """1 step 分の CWGD 変調を適用する。

        Args:
            inputs: 単一 Tensor、または (x_num, x_cat) のような tuple/list。
            forward_fn: Hutchinson refresh (実モデルでの forward+backward) で
                使う `forward_fn(model, *inputs) -> outputs`。None なら
                `model(*inputs)`。
            per_sample_loss_fn: per-sample 勾配計算 (vmap) で使う
                `per_sample_loss_fn(model, params, buffers, *sample_args, target) -> loss`。
                None なら `functional_call` を使った既定実装 (単一 Tensor
                入力向け)。FT-Transformer 等では必須。
                詳しくは _compute_cwgd_t / refresh_hutchinson の docstring 参照。

        呼び出し前提: `loss.backward()` は直前に呼ばれているが、この関数は
        model.parameters() の `.grad` を上書きする一時的な forward/backward
        を内部で行う (最後に `.grad` を元通り復元して返す)。呼び出し側は
        この関数の後で通常どおり `optimizer.step()` を呼べばよい。
        """
        self.step_count += 1

        # 呼び出し側の .grad を保存 (per-sample grad 計算 / Hutchinson refresh の
        # 副作用として .grad が上書きされるのを防ぐため)
        saved_grad = [p.grad.detach().clone() if p.grad is not None else None
                      for p in model.parameters()]

        need_refresh = (
            self.lambda_diag is None
            or (self.step_count - (self.last_hutchinson_refresh_step or 0))
            >= self.refresh_interval_steps
        )
        if need_refresh:
            self.refresh_hutchinson(model, criterion, inputs, labels, forward_fn=forward_fn)

        cwgd_t = self._compute_cwgd_t(model, criterion, inputs, labels,
                                       per_sample_loss_fn=per_sample_loss_fn)
        if self.cwgd0 is None or self.cwgd0 <= 0:
            self.cwgd0 = max(cwgd_t, 1e-12)

        mult = 1.0 / (1.0 + self.alpha * (cwgd_t / self.cwgd0))
        prev_mult = self.current_mult
        self.current_mult = mult
        self.last_cwgd_t = cwgd_t
        self.last_mult = mult

        # .grad を復元
        for p, g in zip(model.parameters(), saved_grad):
            if g is None:
                p.grad = None
            elif p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

        # ----- CosineAnnealingLR の base_lrs を変調 (CTRL と同じパターン) -----
        if self.original_base_lrs is None and scheduler is not None and hasattr(scheduler, "base_lrs"):
            self.original_base_lrs = list(scheduler.base_lrs)

        if scheduler is not None and hasattr(scheduler, "base_lrs") and self.original_base_lrs is not None:
            scheduler.base_lrs = [ob * mult for ob in self.original_base_lrs]
            # 現在の pg["lr"] にも即時反映 (ctrl_v5_continuous と同じ「prev_mult 比で
            # 補正する」方式。単純に pg["lr"] *= mult すると同一 epoch 内の step ごとに
            # 多重に乗算されてしまうため、prev_mult との比で補正する必要がある)
            if prev_mult > 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["lr"] * (mult / prev_mult)

        return {
            "step": self.step_count,
            "cwgd_t": cwgd_t,
            "cwgd_0": self.cwgd0,
            "mult": mult,
            "hutchinson_refreshed": need_refresh,
            "lambda_mean_abs": (
                float(torch.cat([l.reshape(-1) for l in self.lambda_diag]).mean().item())
                if self.lambda_diag is not None else None
            ),
        }
