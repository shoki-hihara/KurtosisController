"""
評価指標確認用スクリプト (v6診断実験: SGD差し替え診断 専用版)

このファイルは v6_diagnostics/ 配下の SGD差し替え診断実験
(train_tabular_sgd_diag.py, optimizer_override=sgd) 専用に、
Axis B本実験の評価スクリプト eval_tabular_wandb.py (v3) を
最小限アレンジしたものです。project_ctrl_v6_design.md 参照。

eval_tabular_wandb.py (v3) からの変更点 (このファイル固有):
  - project_for() が返すW&Bプロジェクト名に "-sgddiag" suffixを追加。
    (train_tabular_sgd_diag.py は optimizer_override が None でない場合、
    プロジェクト名を f"...-{protocol}-sgddiag" とするため。この修正なしだと
    Axis B本実験(AdamW)の"古い"結果を静かに再取得・表示してしまう
    重大なバグになるので必ず本ファイルを使うこと。)
  - RUN_NAME_RE の末尾に (?:-sgdopt)? を追加。(train_tabular_sgd_diag.py の
    run_name は optimizer_override指定時に末尾へ "-sgdopt" が付与されるため、
    元の正規表現(完全一致)だと該当runが全て「命名パターン不一致」として
    スキップされてしまう。)
  - DATASETS を今回実際に実行した covtype のみに限定。
  - METHODS を今回実際に実行した ["CTRL", "COSINE"] のみに限定。
  - extract_run_metrics() に "grad/kurtosis" (生のk_t) の最小値
    (kurtosis_min) を追加抽出し、自動診断セクションで
    baseline<=0 または kurtosis_min<-2 (excess kurtosisの理論的下限) を
    明示的に警告するようにした。Axis C (LSTM/PTB) のpilotで見つかったのと
    同じ症状で、原因は train_tabular_sgd_diag.py の excess_kurtosis() に
    あった eps のスケール不整合バグ (v < eps のガードと v**2 + eps の
    分母でepsのスケールが不一致で、結果が-3側に系統的に歪む) と確定して
    おり、★2026-08-30にこのファイル側のexcess_kurtosis()も修正済み
    (project_text_experiments.md / project_ctrl_v6_design.md 参照)。
    修正後もこの警告が出た場合は別の原因を疑う必要があるため、引き続き
    自動診断に残してある。

以下は元の eval_tabular_wandb.py (v3) のdocstring(引き続き有効な説明):

画像実験(fromscratch-ES)用の v2 スクリプトを、表データ実験
(train_tabular_ft_transformer.py, California Housing/Adult/Covertype)に
合わせて全面的に書き換えた。

主な変更点 (vs v2, 画像実験用):
  - 単一 PROJECT / 単一 target(0.75) / 単一 metric列("val/acc") 前提だったのを、
    California Housing(regression, RMSE, 低いほど良い) / Adult・Covertype
    (binclass・multiclass, accuracy, 高いほど良い) の3データセット・
    task_type別に metric列・reach方向・reach閾値を切り替えられるよう一般化した。
    (train_tabular_ft_transformer.py の TASK_METRIC_HIGHER_IS_BETTER /
    DATASET_CONFIGS[dataset]["reach_threshold"] と同じ値・同じ方向性で判定する)
  - PROJECTがデータセットごとに別プロジェクトになっている
    (`crazy-sonnet-ambl/KurtosisEWMController-tabular-<dataset>-<protocol>`)
    ため、データセットごとにprojectを切り替えてまとめて評価する構成にした。
  - runs_dict を手動で埋める方式をやめ、run名のパターン
    ("{dataset}-fttransformer-{protocol}-{scheduler}-seed{seed}",
    train_tabular_ft_transformer.py の run_name と同一) から**自動的に
    run IDを発見する**方式に変更した(6手法×3seed×3データセット=54 runを
    手打ちするのは非現実的なため)。
  - W&B configの"method"フィールドはCTRL run・COSINE runとも"COSINE"と誤って
    記録される既知の問題があるため(train_tabular_ft_transformer.py側の
    'method'ログ自体の軽微な不整合。詳細は project_tabular_experiments.md 参照)、
    method判定はconfigではなく**run名から抽出したscheduler名**を正とする。
  - v2にあった `control/target_mult` / `control/action_code` は
    train_tabular_ft_transformer.py では per-step wandb ログに含まれていない
    (ローカルの ctrl_trace CSV にのみ記録される) ため、該当フィールドは削除。
    介入回数(interventions)は `control/interventions` の累積値(正確)を使うが、
    介入が起きた正確なstepの一覧やdecay開始stepは、per-step wandbログが
    log_every=50 step間隔の間引きログである関係で近似値になる点に注意。
    **正確な値が必要な場合は各runの `<dataset>_ctrl_seed<N>_ctrl_trace.csv`
    (全step記録)を直接参照すること。**
  - `train/best_val_epoch` / `train/best_val_metric` / `train/stop_epoch` は
    train_tabular_ft_transformer.py の学習終了時ログ(test評価と同じstepで記録)
    からそのまま取得できるためそちらを使用(v2の"Plan A"相当の情報が
    最初から用意されている)。

使い方:
    python3 eval_sgd_diag_pilot.py > eval_sgd_diag_pilot_report.txt 2>&1
    cat eval_sgd_diag_pilot_report.txt
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

# =====================================================================
# 設定
# =====================================================================

WANDB_ENTITY = "crazy-sonnet-ambl"
PROTOCOL = "noes"
# ★このファイルではsmoke/full実行済みの CTRL/COSINE のみに限定
METHODS = ["CTRL", "COSINE"]

# train_tabular_ft_transformer.py の TASK_METRIC_HIGHER_IS_BETTER /
# DATASET_CONFIGS[dataset]["reach_threshold"] と同じ値・同じ方向性
# ★このファイルではSGD差し替え診断実験で実際に使用した covtype のみに限定
DATASETS = {
    "covtype": dict(
        task_type="multiclass", higher_is_better=True,
        target=0.90,
        val_metric_col="val/acc", test_metric_col="test/acc_final",
    ),
}

TIMESERIES_OUT_DIR = Path("./v5_timeseries_tabular")

# ★末尾に (?:-sgdopt)?(?:-lr[0-9.]+)? を追加: train_tabular_sgd_diag.py は
# optimizer_override指定時、run_name末尾に "-sgdopt" を、さらに --sgd_lr で
# 既定lr(0.1)を上書きした場合は "-lr{lr}" も付与するため(2026-08-30、
# smokeがbest_val_epoch=0で実質学習未進行だったため、複数lrで再検証できるように
# 追加した機能。project_ctrl_v6_design.md参照)。
RUN_NAME_RE = re.compile(
    r"^(?P<dataset>[a-z_]+)-fttransformer-(?P<protocol>[a-z0-9]+)-"
    r"(?P<method>[a-z_]+)-seed(?P<seed>\d+)(?:-sgdopt)?(?:-lr(?P<sgd_lr>[0-9.]+))?$"
)


def project_for(dataset: str, protocol: str = PROTOCOL) -> str:
    # ★SGD差し替え診断実験専用: train_tabular_sgd_diag.py は
    # optimizer_override指定時、プロジェクト名末尾に "-sgddiag" を付与するため
    # (この suffix がないと Axis B本実験(AdamW)の"古い"結果を誤取得してしまう)
    return f"{WANDB_ENTITY}/KurtosisEWMController-tabular-{dataset}-{protocol}-sgddiag"


# =====================================================================
# run自動発見: run名 "{dataset}-fttransformer-{protocol}-{method}-seed{N}" から
# method/seed を復元する (W&B configの'method'フィールドはCTRL/COSINEとも
# "COSINE"と誤記録される既知の問題があるため、config not run名を正とする)
# =====================================================================

def discover_runs(dataset: str, protocol: str = PROTOCOL):
    """{method_upper: {seed: run_id}} を返す。"""
    project = project_for(dataset, protocol)
    api = wandb.Api()
    try:
        runs = list(api.runs(project))
    except Exception as e:
        print(f"[SKIP] {project} にアクセスできませんでした: {e}")
        return {}

    found = {}
    unmatched = []
    for r in runs:
        m = RUN_NAME_RE.match(r.name)
        if not m or m.group("dataset") != dataset or m.group("protocol") != protocol:
            unmatched.append(r.name)
            continue
        method = m.group("method").upper()
        seed = int(m.group("seed"))
        found.setdefault(method, {})[seed] = r.id

    if unmatched:
        print(f"[INFO] {project}: 命名パターンに一致しなかったrun {len(unmatched)}件 "
              f"(古いpilot run等の可能性、無視): {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    n_total = sum(len(v) for v in found.values())
    print(f"[{dataset}] {project} から {n_total} run発見: "
          + ", ".join(f"{m}={len(v)}seed" for m, v in sorted(found.items())))
    return found


# =====================================================================
# ヘルパ
# =====================================================================

def _to_numeric_series(df, col):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return None
    return s


def _safe_last(df, col):
    s = _to_numeric_series(df, col)
    return float(s.iloc[-1]) if s is not None else None


def _safe_mean(df, col):
    s = _to_numeric_series(df, col)
    return float(s.mean()) if s is not None else None


def _safe_max(df, col):
    s = _to_numeric_series(df, col)
    return float(s.max()) if s is not None else None


def _safe_min(df, col):
    s = _to_numeric_series(df, col)
    return float(s.min()) if s is not None else None


# =====================================================================
# 1 run 分の抽出
# =====================================================================

def extract_run_metrics(run_id, project, *, val_metric_col, test_metric_col,
                         target, higher_is_better,
                         save_timeseries=False, ts_out_dir=None):
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")

    rows = list(run.scan_history())
    df = pd.DataFrame(rows)

    if "_step" not in df.columns:
        print(f"[SKIP] {run_id}: _step missing")
        return None

    df = df.sort_values("_step").reset_index(drop=True)

    # val metric 履歴 (epoch 単位)
    df_val = run.history(keys=["epoch", val_metric_col], pandas=True)
    if df_val is None or len(df_val) == 0:
        print(f"[SKIP] {run_id}: validation history ({val_metric_col}) missing")
        return None

    df_val = (
        df_val
        .dropna(subset=["epoch", val_metric_col])
        .sort_values("epoch")
        .drop_duplicates("epoch", keep="last")
        .reset_index(drop=True)
    )
    if len(df_val) == 0:
        print(f"[SKIP] {run_id}: val history empty after clean")
        return None

    final_row = df_val.iloc[-1]
    final_metric = float(final_row[val_metric_col])

    stop_epoch = None
    if "epoch" in df_val.columns and pd.notna(final_row.get("epoch")):
        stop_epoch = int(final_row["epoch"])
    if "train/stop_epoch" in df.columns:
        se_rows = df.dropna(subset=["train/stop_epoch"])
        if len(se_rows) > 0:
            stop_epoch = int(se_rows["train/stop_epoch"].iloc[-1])

    # reach 判定 (higher_is_better に応じて方向を切り替える。
    # train_tabular_ft_transformer.py の
    # `reached = (val_metric >= reach_threshold) if higher_is_better else (val_metric <= reach_threshold)`
    # と同じロジック)
    if higher_is_better:
        hit = df_val[df_val[val_metric_col] >= target]
    else:
        hit = df_val[df_val[val_metric_col] <= target]

    if len(hit) > 0:
        first = hit.iloc[0]
        reach = 1
        epoch_reach = int(first["epoch"])
        val_metric_at_reach = float(first[val_metric_col])
        # margin: 常に「良い方向にどれだけ余裕があるか」が正の値になるよう統一
        # (regressionのようにlower-is-betterでも解釈しやすくするため)
        val_metric_margin = (
            (val_metric_at_reach - target) if higher_is_better
            else (target - val_metric_at_reach)
        )

        step_reach = None
        if "epoch" in df.columns:
            tmp = df[["epoch", "_step"]].dropna().sort_values("_step")
            match = tmp[tmp["epoch"] == epoch_reach]
            if len(match) > 0:
                step_reach = int(match["_step"].max())

        target_epochs = list(range(epoch_reach - 2, epoch_reach + 3))
        nbrs = df_val[df_val["epoch"].isin(target_epochs)].sort_values("epoch")
        val_metric_neighbors = (
            {int(r["epoch"]): float(r[val_metric_col]) for _, r in nbrs.iterrows()}
            if len(nbrs) > 0 else None
        )
    else:
        reach = 0
        epoch_reach = None
        step_reach = None
        val_metric_at_reach = final_metric
        val_metric_margin = (
            (final_metric - target) if higher_is_better else (target - final_metric)
        )
        val_metric_neighbors = None

    # final_test_metric: best-val checkpointで評価したtest指標
    final_test_metric = None
    if test_metric_col in df.columns:
        test_rows = df.dropna(subset=[test_metric_col])
        if len(test_rows) > 0:
            final_test_metric = float(test_rows[test_metric_col].iloc[-1])

    # best_val_metric / best_val_epoch (train_tabular_ft_transformer.py が
    # 学習終了時に直接ログしているのでそのまま取得できる)
    best_val_metric = None
    best_val_epoch = None
    if "train/best_val_metric" in df.columns:
        bvm_rows = df.dropna(subset=["train/best_val_metric"])
        if len(bvm_rows) > 0:
            best_val_metric = float(bvm_rows["train/best_val_metric"].iloc[-1])
    if "train/best_val_epoch" in df.columns:
        bve_rows = df.dropna(subset=["train/best_val_epoch"])
        if len(bve_rows) > 0:
            best_val_epoch = int(bve_rows["train/best_val_epoch"].iloc[-1])

    # v5 コントローラ状態指標 (CTRL のみ有効)
    baseline_val = _safe_last(df, "control/baseline")
    # ★Axis Cで見つかった数値バグ(baseline<=0 かつ k_t<-2という理論的に
    # 不可能な値)の症状チェック用。excess kurtosisの理論的下限は-2。
    kurtosis_min = _safe_min(df, "grad/kurtosis")
    current_mult_final = _safe_last(df, "control/current_mult")
    current_mult_min = _safe_min(df, "control/current_mult")
    current_mult_mean = _safe_mean(df, "control/current_mult")
    ratio_mean = _safe_mean(df, "control/ratio")
    ratio_max = _safe_max(df, "control/ratio")

    ratio_p95 = None
    if "control/ratio" in df.columns:
        ratio_series = pd.to_numeric(df["control/ratio"], errors="coerce").dropna()
        if len(ratio_series) > 0:
            ratio_p95 = float(ratio_series.quantile(0.95))

    interventions = _safe_last(df, "control/interventions")
    if interventions is not None:
        interventions = int(interventions)

    # ★注意: per-step wandbログは log_every=50 step間隔の間引きログのため、
    # 以下の intervention_steps / decay_start_step は近似値(±50 step程度の誤差)。
    # 正確な値はローカルの ctrl_trace CSV を参照すること。
    intervention_steps_approx = None
    if "control/current_mult" in df.columns:
        cm = df[["_step", "control/current_mult"]].dropna().sort_values("_step")
        if len(cm) > 1:
            drops = cm["control/current_mult"].diff()
            decay_rows = cm[drops < -1e-4]
            if len(decay_rows) > 0:
                intervention_steps_approx = decay_rows["_step"].astype(int).tolist()

    decay_start_step_approx = None
    if "control/current_mult" in df.columns:
        cm = df[["_step", "control/current_mult"]].dropna()
        below_one = cm[cm["control/current_mult"] < 0.9999]
        if len(below_one) > 0:
            decay_start_step_approx = int(below_one.sort_values("_step").iloc[0]["_step"])

    # LR 軌道
    lr_initial = lr_at_reach = lr_final = None
    if "lr" in df.columns:
        df_lr = df.dropna(subset=["lr"]).sort_values("_step")
        if len(df_lr) > 0:
            lr_initial = float(df_lr["lr"].iloc[0])
            lr_final = float(df_lr["lr"].iloc[-1])
            if step_reach is not None:
                lr_before = df_lr[df_lr["_step"] <= step_reach]
                if len(lr_before) > 0:
                    lr_at_reach = float(lr_before["lr"].iloc[-1])

    # VRAM実測 (train_tabular_ft_transformer.py 固有の追加ログ)
    peak_mem_alloc_mb = _safe_last(df, "gpu/peak_memory_allocated_mb")
    peak_mem_reserved_mb = _safe_last(df, "gpu/peak_memory_reserved_mb")

    if save_timeseries and ts_out_dir is not None:
        import pickle
        ts_out_dir = Path(ts_out_dir)
        ts_out_dir.mkdir(parents=True, exist_ok=True)
        ts_data = {
            "run_id": run_id,
            "_step": df["_step"].tolist() if "_step" in df.columns else [],
        }
        for col in ["control/current_mult", "control/ratio", "control/baseline",
                    "grad/kurtosis", "grad/kurtosis_ewm",
                    "lr", "train/loss", "epoch"]:
            if col in df.columns:
                ts_data[col] = df[col].tolist()
        if len(df_val) > 0:
            ts_data[f"{val_metric_col}_epoch"] = df_val["epoch"].astype(int).tolist()
            ts_data[val_metric_col] = df_val[val_metric_col].tolist()

        out_path = ts_out_dir / f"{run_id}_timeseries.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(ts_data, f)

    return {
        "reach": reach,
        "epoch_reach": epoch_reach,
        "step_reach": step_reach,
        "stop_epoch": stop_epoch,
        "final_metric": final_metric,
        "final_test_metric": final_test_metric,
        "best_val_metric": best_val_metric,
        "best_val_epoch": best_val_epoch,
        "val_metric_at_reach": val_metric_at_reach,
        "val_metric_margin": val_metric_margin,
        "val_metric_neighbors": val_metric_neighbors,
        "baseline": baseline_val,
        "kurtosis_min": kurtosis_min,
        "current_mult_final": current_mult_final,
        "current_mult_min": current_mult_min,
        "current_mult_mean": current_mult_mean,
        "ratio_mean": ratio_mean,
        "ratio_max": ratio_max,
        "ratio_p95": ratio_p95,
        "interventions": interventions,
        "intervention_steps_approx": intervention_steps_approx,
        "decay_start_step_approx": decay_start_step_approx,
        "lr_initial": lr_initial,
        "lr_at_reach": lr_at_reach,
        "lr_final": lr_final,
        "peak_memory_allocated_mb": peak_mem_alloc_mb,
        "peak_memory_reserved_mb": peak_mem_reserved_mb,
        "num_val_records": len(df_val),
    }


# =====================================================================
# 実行 (3データセット分をまとめて評価)
# =====================================================================

def run_evaluation(datasets=DATASETS, protocol=PROTOCOL,
                    save_timeseries=False, ts_out_dir=None):
    results = []
    detail_rows = []

    for dataset, dcfg in datasets.items():
        project = project_for(dataset, protocol)
        runs_by_method = discover_runs(dataset, protocol)

        for method in METHODS:
            seed_map = runs_by_method.get(method, {})
            if not seed_map:
                print(f"  [WARN] {dataset}/{method}: run未発見(pilot未実行 or 命名不一致)")
                continue
            for seed in sorted(seed_map):
                rid = seed_map[seed]
                print(f"  fetching {dataset}/{method} seed={seed} ({rid}) ...", end=" ", flush=True)
                metrics = extract_run_metrics(
                    rid, project,
                    val_metric_col=dcfg["val_metric_col"],
                    test_metric_col=dcfg["test_metric_col"],
                    target=dcfg["target"],
                    higher_is_better=dcfg["higher_is_better"],
                    save_timeseries=save_timeseries, ts_out_dir=ts_out_dir,
                )
                if metrics is None:
                    print("SKIP")
                    continue
                print("OK")

                iv_steps = metrics.pop("intervention_steps_approx")
                nbr = metrics.pop("val_metric_neighbors")

                row = {
                    "dataset": dataset,
                    "task_type": dcfg["task_type"],
                    "method": method,
                    "seed": seed,
                    "run_id": rid,
                    **metrics,
                }
                results.append(row)
                detail_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "run_id": rid,
                    "intervention_steps_approx": iv_steps,
                    "val_metric_neighbors": nbr,
                })

    df_all = pd.DataFrame(results)
    df_detail = pd.DataFrame(detail_rows)
    return df_all, df_detail


# =====================================================================
# 表示
# =====================================================================

def display_results(df_all, df_detail):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", "{:.4f}".format)

    if len(df_all) == 0:
        print("[WARN] df_all is empty — run未発見の可能性")
        return

    print("\n========== 1. Core 指標 ==========")
    cols = [
        "dataset", "method", "seed", "run_id",
        "reach", "epoch_reach",
        "best_val_epoch", "best_val_metric",
        "stop_epoch",
        "final_test_metric",
        "final_metric",
        "interventions",
    ]
    avail = [c for c in cols if c in df_all.columns]
    print(df_all[avail].to_string(index=False))

    print("\n========== 2. v5 状態指標 (CTRL のみ有効) ==========")
    cols = ["dataset", "method", "seed",
            "baseline", "kurtosis_min",
            "current_mult_final", "current_mult_min", "current_mult_mean",
            "ratio_mean", "ratio_max", "ratio_p95",
            "decay_start_step_approx"]
    avail = [c for c in cols if c in df_all.columns]
    print(df_all[avail].to_string(index=False))

    print("\n========== 3. LR 軌道 / VRAM ==========")
    cols = ["dataset", "method", "seed", "lr_initial", "lr_at_reach", "lr_final",
            "peak_memory_allocated_mb"]
    avail = [c for c in cols if c in df_all.columns]
    print(df_all[avail].to_string(index=False))

    print("\n========== 4. 介入 step リスト (CTRL のみ、近似値・±50step程度) ==========")
    for _, r in df_detail.iterrows():
        if r["method"] != "CTRL":
            continue
        iv = r["intervention_steps_approx"]
        n = len(iv) if iv is not None else 0
        print(f"  {r['dataset']} {r['method']} seed={r['seed']} ({r['run_id']}): 約{n} 回検出")
        if iv is not None and len(iv) > 0:
            head = iv[:6]
            tail = iv[-3:] if len(iv) > 6 else []
            tail_str = f" ... {tail}" if tail else ""
            print(f"    first 6: {head}{tail_str}")

    print("\n========== 5. val_metric neighbors (reach周辺) ==========")
    for _, r in df_detail.iterrows():
        nbr = r["val_metric_neighbors"]
        if nbr is None:
            continue
        run_id = r["run_id"]
        er_row = df_all.loc[df_all["run_id"] == run_id, "epoch_reach"]
        er = er_row.iloc[0] if len(er_row) > 0 else None
        print(f"  {r['dataset']} {r['method']} seed={r['seed']} ({run_id}):")
        for ep in sorted(nbr.keys()):
            marker = "  <- reach" if er is not None and ep == er else ""
            print(f"    epoch={ep}  metric={nbr[ep]:.4f}{marker}")

    print("\n========== 6. 自動診断 (CTRL) ==========")
    for _, r in df_all.iterrows():
        if r["method"] != "CTRL":
            continue
        label = f"{r['dataset']} {r['method']} seed={r['seed']} ({r['run_id']})"
        print(f"\n  [{label}]")

        # ★2026-08-29修正: extract_run_metrics()はPythonのNoneを返すが、
        # pd.DataFrame(results)に集約された時点でNoneは数値列ではNaNに
        # 変換される(pandasの仕様)。`NaN is None`はFalseになるため、
        # 元の`is None`/`is not None`チェックは欠損値を一切検出できず、
        # 「x ... が記録されていない」ではなく「info ... ≈ nan」のような
        # 誤解を招く出力になっていた(実データで確認済みのバグ)。
        # section 9 (display_summary)で既に使っていた pd.isna()/pd.notna()
        # による判定に統一する。
        if pd.isna(r["baseline"]):
            print("    x baseline 未確定 (burn-in が完了していない可能性)")
        else:
            print(f"    v baseline = {r['baseline']:.2f}")
            if r["baseline"] <= 0:
                print("    !! 警告: baseline <= 0 です。Axis C (LSTM/PTB pilot) で"
                      "見つかったのと同じ症状です。原因はexcess_kurtosis()の"
                      "epsスケール不整合バグと確定済みで、このファイルが使う"
                      "train_tabular_sgd_diag.pyは2026-08-30に修正済みのはずです。"
                      "それでもこの警告が出る場合、このrunが修正前のコードで"
                      "実行された(再実行が必要)か、別の原因が疑われるので、"
                      "project_text_experiments.md / project_ctrl_v6_design.md "
                      "の該当セクションを確認してください。")

        if pd.notna(r.get("kurtosis_min")) and r["kurtosis_min"] < -2:
            print(f"    !! 警告: kurtosis_min = {r['kurtosis_min']:.3f} < -2 "
                  "(excess kurtosisの理論的下限を下回っています)。"
                  "excess_kurtosis()のepsスケール不整合バグ(Axis Cが確定・"
                  "修正済み)の症状です。修正後のコードで再実行したrunか"
                  "確認してください。")

        cm_final = r["current_mult_final"]
        cm_min = r["current_mult_min"]
        if pd.isna(cm_final):
            print("    x current_mult が記録されていない")
        elif cm_final >= 0.99:
            print(f"    x current_mult 最終値 {cm_final:.3f}: ほぼ無介入")
        elif pd.notna(cm_min) and cm_min <= 0.21:
            print(f"    △ current_mult 最低値 {cm_min:.3f}: min_mult (0.2) に張り付き、過剰減衰の可能性")
        elif 0.3 <= cm_final <= 0.7:
            print(f"    v current_mult 最終値 {cm_final:.3f}: 適切な減衰")
        else:
            print(f"    info current_mult 最終値 {cm_final:.3f}, 最低値 {cm_min}")

        iv = r["interventions"]
        if pd.isna(iv):
            print("    x interventions が記録されていない")
        elif iv == 0:
            print("    x interventions=0: controller が全く動作していない")
        else:
            print(f"    v interventions = {iv} 回")

        if pd.notna(r["ratio_max"]) and pd.notna(r["ratio_mean"]):
            p95_str = f"{r['ratio_p95']:.2f}" if pd.notna(r["ratio_p95"]) else "NaN"
            print(f"    info ratio: mean={r['ratio_mean']:.2f}, max={r['ratio_max']:.2f}, "
                  f"p95={p95_str}")

        if pd.notna(r["decay_start_step_approx"]):
            print(f"    info 最初の介入 step ≈ {r['decay_start_step_approx']} (近似値)")
        elif pd.isna(iv) or iv == 0:
            print("    info 最初の介入 step: 介入が検出されなかったため該当なし")

        # burn-in前にreachしていないかの簡易チェック(Adult/California Housingで
        # 実際に起きた問題。正確なburn-in完了epochはreport_pilot_results.pyの
        # burnin_epoch_equiv()を参照)
        if r["reach"] == 1 and pd.notna(r["epoch_reach"]) and pd.notna(r["best_val_epoch"]):
            if r["best_val_epoch"] < r["epoch_reach"]:
                pass  # 通常はreach <= best_valのはずなので特に警告なし


# =====================================================================
# 集計 (dataset × method の seed 平均)
# =====================================================================

def display_summary(df_all):
    print("\n========== 7. 集計 (dataset x method, seed平均) ==========")
    if len(df_all) == 0:
        print("  [WARN] df_all is empty")
        return

    agg_dict = {
        "n":                    ("seed", "count"),
        "reach_rate":           ("reach", "mean"),
        "epoch_reach_mean":     ("epoch_reach", "mean"),
        "epoch_reach_std":      ("epoch_reach", "std"),
        "best_val_metric_mean": ("best_val_metric", "mean"),
        "best_val_metric_std":  ("best_val_metric", "std"),
        "best_val_epoch_mean":  ("best_val_epoch", "mean"),
        "stop_epoch_mean":      ("stop_epoch", "mean"),
        "stop_epoch_std":       ("stop_epoch", "std"),
        "final_test_metric_mean": ("final_test_metric", "mean"),
        "final_test_metric_std":  ("final_test_metric", "std"),
        "cm_final_mean":        ("current_mult_final", "mean"),
        "ratio_mean_mean":      ("ratio_mean", "mean"),
    }
    agg_dict = {k: v for k, v in agg_dict.items() if v[0] in df_all.columns}

    grp = df_all.groupby(["dataset", "method"]).agg(**agg_dict)
    print(grp.to_string())

    print("\n========== 8. 主要指標比較 (論文記載候補) ==========")
    key_cols = ["final_test_metric_mean", "final_test_metric_std",
                "best_val_metric_mean", "epoch_reach_mean", "epoch_reach_std"]
    avail = [c for c in key_cols if c in grp.columns]
    print(grp[avail].to_string())

    # CTRL vs 各baselineの差がseed間ばらつきに埋もれるかの記述的判定
    # (report_pilot_results.py の check_multiseed_robustness と同じ考え方)
    print("\n========== 9. CTRL vs baseline (seed間ばらつきとの比較、記述的目安) ==========")
    for dataset in df_all["dataset"].unique():
        sub = df_all[df_all["dataset"] == dataset]
        g = sub.groupby("method")["final_test_metric"].agg(["count", "mean", "std"])
        if "CTRL" not in g.index:
            continue
        print(f"  [{dataset}]")
        if g.loc["CTRL", "count"] <= 1:
            print(f"    [INFO] CTRLがseed1つのみのため頑健性判定はできません(--seeds 0 1 2等で複数実行してください)。")
            continue
        ctrl_mean = g.loc["CTRL", "mean"]
        ctrl_std = g.loc["CTRL", "std"]
        ctrl_std = 0.0 if pd.isna(ctrl_std) else ctrl_std
        for method in g.index:
            if method == "CTRL":
                continue
            if g.loc[method, "count"] <= 1:
                print(f"    CTRL vs {method}: {method}がseed1つのみのため判定できません。")
                continue
            base_mean = g.loc[method, "mean"]
            base_std = g.loc[method, "std"]
            base_std = 0.0 if pd.isna(base_std) else base_std
            diff = abs(ctrl_mean - base_mean)
            larger_std = max(ctrl_std, base_std)
            verdict = (
                "区別つかず" if (larger_std > 0 and diff < larger_std)
                else ("std=0(完全一致)" if larger_std == 0 and diff == 0 else "有意そう")
            )
            print(f"    CTRL vs {method}: |diff|={diff:.6f}  "
                  f"(std: CTRL={ctrl_std:.6f}, {method}={base_std:.6f})  → {verdict}")


# =====================================================================
# main
# =====================================================================

if __name__ == "__main__":
    df_all, df_detail = run_evaluation(
        DATASETS, PROTOCOL,
        save_timeseries=False,
    )
    display_results(df_all, df_detail)
    display_summary(df_all)
