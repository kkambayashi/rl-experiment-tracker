"""条件ごとの集計。

「条件 (condition)」= アルゴリズム + 実験間で値が異なるハイパーパラメータの組み合わせ。
seed は条件に含めない (同じ条件を複数 seed で回したものを 1 グループに数えたいため)。

数値計算は pandas に頼らず標準ライブラリで書いている。
テストで期待値を手計算しやすく、UI 以外からも使えるようにするため。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable

from .models import DEFAULT_SCORE_METRIC, STATUS_FINISHED, ExperimentRun

# 条件の識別に使わないキー。
#  - server_address: 研究室の内部 IP。結果には無関係で、表示もしない
#  - seed: 条件ではなく「同一条件の繰り返し」を区別する物
#  - agent.type: algorithm として常に条件キーの先頭に入るので重複
#  - metadata.name: 使うメタデータ名のリスト。個々の値 (global_eta 等) が別キーで入るので冗長
DEFAULT_IGNORED_KEYS: frozenset[str] = frozenset({
    "worker_args.server_address",
    "train_args.seed",
    "train_args.agent.type",
    "train_args.metadata.name",
})

# パラメータが無い実験を表す目印 (None は「値が None」なので区別する)
MISSING = "—"


# --------------------------------------------------------------------------
# 条件キー
# --------------------------------------------------------------------------
def varying_parameters(
    runs: Iterable[ExperimentRun],
    ignored: frozenset[str] = DEFAULT_IGNORED_KEYS,
    include_presence: bool = False,
) -> list[str]:
    """読み込んだ実験の間で値が 1 種類でないパラメータ名を返す (辞書順)。

    include_presence=False (既定): そのキーを持つ実験同士で値が異なる物だけ。
        例: global_eta が 0.1 と 0.5 → 含む。c_floor が一部の実験にしか無いが値は 0.001 のみ → 含まない。
        「キーの有無」はアルゴリズムの違い (BASE には metadata が無い等) に由来することが多く、
        条件ラベルに全部出すと読めなくなるため既定では外す。
    include_presence=True: 一部の実験にしか無いキーも「異なる」とみなす (欠損は MISSING)。
    """
    runs = list(runs)
    all_keys: set[str] = set()
    for r in runs:
        all_keys.update(r.parameters)

    varying: list[str] = []
    for key in sorted(all_keys - ignored):
        if include_presence:
            values = {_hashable(r.parameters.get(key, MISSING)) for r in runs}
        else:
            values = {_hashable(r.parameters[key]) for r in runs if key in r.parameters}
        if len(values) > 1:
            varying.append(key)
    return varying


def condition_key(run: ExperimentRun, keys: list[str]) -> tuple[Hashable, ...]:
    """グループ化に使うタプル。先頭はアルゴリズム、以降は keys の値 (無ければ MISSING)。"""
    return (run.algorithm or MISSING,) + tuple(
        _hashable(run.parameters.get(k, MISSING)) for k in keys
    )


def condition_label(run: ExperimentRun, keys: list[str]) -> str:
    """人が読むためのラベル。例: 'SRS_v2_v3 | global_eta=0.2 | c_floor=—'"""
    parts = [str(run.algorithm or MISSING)]
    for k in keys:
        parts.append(f"{short_key(k)}={run.parameters.get(k, MISSING)}")
    return " | ".join(parts)


def short_key(key: str) -> str:
    """'train_args.metadata.global_eta' -> 'global_eta' のように、読みやすい短い名前にする。"""
    for prefix in ("train_args.metadata.", "train_args.agent.", "train_args.", "env_args.param.", "env_args.", "worker_args."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def group_by_condition(
    runs: Iterable[ExperimentRun], keys: list[str]
) -> dict[tuple[Hashable, ...], list[ExperimentRun]]:
    groups: dict[tuple[Hashable, ...], list[ExperimentRun]] = {}
    for r in runs:
        groups.setdefault(condition_key(r, keys), []).append(r)
    return groups


# --------------------------------------------------------------------------
# 統計
# --------------------------------------------------------------------------
def mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    """None を除いた平均・標本標準偏差 (ddof=1)・件数。

    n == 0 -> (None, None, 0)
    n == 1 -> (平均, None, 1)   ※ 1 件では標準偏差を定義しない (0 と誤読させない)
    """
    xs = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(xs)
    if n == 0:
        return None, None, 0
    mean = sum(xs) / n
    if n == 1:
        return mean, None, 1
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var), n


@dataclass
class ConditionSummary:
    key: tuple[Hashable, ...]
    label: str
    algorithm: str
    parameters: dict[str, Any]          # 条件を構成するパラメータ (short_key -> 値)
    n_runs: int
    n_finished: int
    seeds: list[int | None]             # 使用済み seed (重複除去・順序保持)
    batches: list[str]                  # どのバッチに含まれるか
    metric: str
    mean: float | None
    std: float | None
    n_values: int                       # 平均に使えた件数 (final 値が取れたラン)
    min: float | None
    max: float | None
    run_ids: list[str] = field(default_factory=list)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)


def summarize_conditions(
    runs: Iterable[ExperimentRun],
    keys: list[str],
    metric: str = DEFAULT_SCORE_METRIC,
    include_incomplete: bool = False,
) -> list[ConditionSummary]:
    """条件ごとに実行回数・seed・最終値の平均と標準偏差をまとめる。

    集計対象の最終値は run.final_value(metric)。
    未完了ランは n_runs には数えるが、既定では平均・標準偏差から除外する
    (途中で止まったランの「最後の値」を最終評価値として混ぜると結果を歪めるため)。
    include_incomplete=True で含められる。何件が平均に入ったかは n_values で分かる。
    """
    summaries: list[ConditionSummary] = []
    for key, group in group_by_condition(runs, keys).items():
        scored = group if include_incomplete else [r for r in group if r.status == STATUS_FINISHED]
        finals = [r.final_value(metric) for r in scored]
        mean, std, n_values = mean_std(finals)
        present = [f for f in finals if f is not None]
        first = group[0]
        summaries.append(
            ConditionSummary(
                key=key,
                label=condition_label(first, keys),
                algorithm=str(first.algorithm or MISSING),
                parameters={short_key(k): first.parameters.get(k, MISSING) for k in keys},
                n_runs=len(group),
                n_finished=sum(1 for r in group if r.status == STATUS_FINISHED),
                seeds=_unique([r.seed for r in group]),
                batches=_unique([r.batch_id for r in group]),
                metric=metric,
                mean=mean,
                std=std,
                n_values=n_values,
                min=min(present) if present else None,
                max=max(present) if present else None,
                run_ids=[r.run_id for r in group],
            )
        )
    summaries.sort(key=lambda s: s.label)
    return summaries


def mean_curve(
    runs: Iterable[ExperimentRun], metric: str
) -> tuple[list[int], list[float | None], list[float | None], list[int]]:
    """複数ランの学習曲線をエポック位置ごとに平均する。

    ランごとに長さが違う (途中で止まった) 場合は、各エポック位置で値のあるランだけで
    平均を取る。返り値の n[i] を見れば、そのエポックが何本のランから計算されたか分かる。
    """
    series = [r.metrics.get(metric) or [] for r in runs]
    length = max((len(s) for s in series), default=0)
    epochs, means, stds, counts = [], [], [], []
    for i in range(length):
        column = [s[i] for s in series if i < len(s)]
        m, sd, n = mean_std(column)
        epochs.append(i)
        means.append(m)
        stds.append(sd)
        counts.append(n)
    return epochs, means, stds, counts


# --------------------------------------------------------------------------
# 比較用
# --------------------------------------------------------------------------
def parameter_diff(
    runs: list[ExperimentRun],
    ignored: frozenset[str] = DEFAULT_IGNORED_KEYS,
) -> dict[str, dict[str, Any]]:
    """選択したラン同士で値が異なるパラメータだけを返す。

    比較画面用なので「片方にしか無いキー」も差分として含める (include_presence=True)。
    返り値: {パラメータ名: {run_id: 値 or MISSING}}
    """
    keys = varying_parameters(runs, ignored, include_presence=True)
    return {k: {r.run_id: r.parameters.get(k, MISSING) for r in runs} for k in keys}


def missing_parameters(runs: list[ExperimentRun]) -> dict[str, list[str]]:
    """比較対象の一部にしか存在しないパラメータと、それを持たないランの一覧。

    「比較が公平か」を判断する材料。例: c_floor を持つ実験と持たない実験を比べていないか。
    """
    all_keys: set[str] = set()
    for r in runs:
        all_keys.update(r.parameters)
    missing: dict[str, list[str]] = {}
    for key in sorted(all_keys - DEFAULT_IGNORED_KEYS):
        absent = [r.run_id for r in runs if key not in r.parameters]
        if absent and len(absent) < len(runs):
            missing[key] = absent
    return missing


# --------------------------------------------------------------------------
# バッチ単位
# --------------------------------------------------------------------------
def batch_overview(
    runs: Iterable[ExperimentRun], keys: list[str]
) -> dict[str, list[tuple[str, int]]]:
    """バッチごとに「含まれる条件ラベルとラン数」を返す。図の横に文脈として出すための物。

    返り値: {batch_id: [(条件ラベル, ラン数), ...]}  (条件ラベル順)
    """
    per_batch: dict[str, dict[str, int]] = {}
    for r in runs:
        label = condition_label(r, keys)
        counts = per_batch.setdefault(r.batch_id, {})
        counts[label] = counts.get(label, 0) + 1
    return {b: sorted(c.items()) for b, c in per_batch.items()}


# --------------------------------------------------------------------------
def _hashable(value: Any) -> Hashable:
    if isinstance(value, (list, dict, set)):
        return repr(value)
    return value


def _unique(values: Iterable[Any]) -> list[Any]:
    seen: list[Any] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen
