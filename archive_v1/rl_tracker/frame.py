"""ExperimentRun の list を pandas.DataFrame に変換する (Streamlit 表示用)。

ここには計算ロジックを置かない。計算は aggregate.py、表示の形にするのがこのモジュール。
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .aggregate import MISSING, ConditionSummary, condition_label, mean_curve, short_key
from .models import DEFAULT_SCORE_METRIC, ExperimentRun

# 一覧の固定列。ハイパーパラメータ列はこの間に動的に挿入する
_FIXED_HEAD = ["run_id", "algorithm"]
_FIXED_TAIL = ["seed", "started_at", "final_score", "status", "epochs", "episodes", "duration_min", "source_path"]


def runs_to_frame(
    runs: Iterable[ExperimentRun],
    param_keys: list[str],
    score_metric: str = DEFAULT_SCORE_METRIC,
) -> pd.DataFrame:
    """実験一覧テーブル。param_keys で指定したハイパーパラメータを列として並べる。"""
    rows = []
    for r in runs:
        row = {
            "run_id": r.run_id,
            "algorithm": r.algorithm,
        }
        for k in param_keys:
            # 0.1 (float) と "—" (欠損) が同じ列に混ざると Arrow 変換で警告が出るので文字列に揃える
            row[short_key(k)] = _display(r.parameters.get(k, MISSING))
        row.update({
            "seed": r.seed,
            "started_at": r.started_at,
            "final_score": r.final_value(score_metric),
            "status": r.status,
            "epochs": r.epochs,
            "episodes": r.episodes,
            "duration_min": round(r.duration_sec / 60, 1) if r.duration_sec is not None else None,
            "source_path": r.source_path,
            "warnings": len(r.warnings),
        })
        rows.append(row)

    columns = _FIXED_HEAD + [short_key(k) for k in param_keys] + _FIXED_TAIL + ["warnings"]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
        for col in ("seed", "final_score", "epochs", "episodes", "duration_min"):
            df[col] = pd.to_numeric(df[col], errors="coerce")  # None 混じりでも数値列にする
    return df


def curves_to_frame(
    runs: Iterable[ExperimentRun],
    metric: str,
    condition_keys: list[str],
) -> pd.DataFrame:
    """学習曲線の long 形式: run_id, condition, epoch, value"""
    rows = []
    for r in runs:
        label = condition_label(r, condition_keys)
        for i, v in enumerate(r.metrics.get(metric) or []):
            if v is not None:
                rows.append({"run_id": r.run_id, "condition": label, "epoch": i, "value": v})
    return pd.DataFrame(rows, columns=["run_id", "condition", "epoch", "value"])


def mean_curves_to_frame(
    groups: dict[str, list[ExperimentRun]],
    metric: str,
) -> pd.DataFrame:
    """条件ごとの平均曲線: condition, epoch, mean, std, lower, upper, n"""
    rows = []
    for label, group in groups.items():
        epochs, means, stds, counts = mean_curve(group, metric)
        for e, m, sd, n in zip(epochs, means, stds, counts):
            if m is None:
                continue
            band = sd if sd is not None else 0.0
            rows.append({
                "condition": label, "epoch": e, "mean": m, "std": sd,
                "lower": m - band, "upper": m + band, "n": n,
            })
    return pd.DataFrame(rows, columns=["condition", "epoch", "mean", "std", "lower", "upper", "n"])


def summaries_to_frame(summaries: list[ConditionSummary]) -> pd.DataFrame:
    """集計画面のテーブル。パラメータ列は条件キーに応じて動的に増える。"""
    rows = []
    param_cols: list[str] = []
    for s in summaries:
        row = {"algorithm": s.algorithm}
        for k, v in s.parameters.items():
            row[k] = _display(v)
            if k not in param_cols:
                param_cols.append(k)
        row.update({
            "n_runs": s.n_runs,
            "n_finished": s.n_finished,
            "n_seeds": s.n_seeds,
            "seeds": ", ".join(str(x) for x in s.seeds),
            "mean": s.mean,
            "std": s.std,
            "min": s.min,
            "max": s.max,
            "n_values": s.n_values,
            "batches": ", ".join(s.batches),
            "label": s.label,
        })
        rows.append(row)
    columns = ["algorithm"] + param_cols + [
        "n_runs", "n_finished", "n_seeds", "seeds", "mean", "std", "min", "max", "n_values", "batches", "label",
    ]
    df = pd.DataFrame(rows, columns=columns)
    # 全て None の数値列が object 型になるのを防ぐ (グラフ側で引き算するため)
    for col in ("mean", "std", "min", "max"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _display(value) -> str:
    """表示用に文字列化。bool は 'True'/'False'、float はそのまま str()。"""
    return str(value)
