"""実験一覧の絞り込み。UI から渡される選択値 (文字列) で ExperimentRun をふるいにかける。

値の比較は文字列で行う。UI の multiselect が返すのは文字列であり、
ハイパーパラメータには 0.1 (float) / 'TD-Q' (str) / '[[1, 5]]' (list の repr) が混在するため、
型を揃えるより「表示している文字列と同じ物を選ぶ」方が確実。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .aggregate import MISSING
from .models import ExperimentRun


@dataclass
class RunFilter:
    algorithms: set[str] = field(default_factory=set)   # 空 = 絞り込まない
    seeds: set[str] = field(default_factory=set)
    statuses: set[str] = field(default_factory=set)
    batches: set[str] = field(default_factory=set)
    parameters: dict[str, set[str]] = field(default_factory=dict)  # パラメータ名 -> 許可する値 (文字列)

    def matches(self, run: ExperimentRun) -> bool:
        if self.algorithms and str(run.algorithm) not in self.algorithms:
            return False
        if self.seeds and str(run.seed) not in self.seeds:
            return False
        if self.statuses and run.status not in self.statuses:
            return False
        if self.batches and run.batch_id not in self.batches:
            return False
        for key, allowed in self.parameters.items():
            if allowed and str(run.parameters.get(key, MISSING)) not in allowed:
                return False
        return True


def apply_filter(runs: Iterable[ExperimentRun], f: RunFilter) -> list[ExperimentRun]:
    return [r for r in runs if f.matches(r)]


def parameter_values(runs: Iterable[ExperimentRun], key: str) -> list[str]:
    """あるパラメータが取る値の一覧 (文字列・重複なし・ソート済み)。無い実験は MISSING。"""
    values = {str(r.parameters.get(key, MISSING)) for r in runs}
    return sorted(values, key=_sort_key)


def _sort_key(s: str):
    """数値に見える文字列は数値順、それ以外は文字列順。MISSING は最後。"""
    if s == MISSING:
        return (2, 0, "")
    try:
        return (0, float(s), "")
    except ValueError:
        return (1, 0, s)
