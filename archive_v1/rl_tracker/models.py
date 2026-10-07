"""アプリ内部で扱うデータ構造。

設計方針:
- 「train_log 1本 = ExperimentRun 1件」とする。
- ハイパーパラメータ (parameters) と評価指標 (metrics) は実験ごとに増減するので、
  固定の列ではなく dict で持つ。表示用の列は後段 (frame.py) で動的に作る。
- ファイルに無い情報は推測せず None のまま残す (seed, git_commit, purpose など)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# 「最終評価値」として既定で使う指標。UI 側で切り替えられる。
DEFAULT_SCORE_METRIC = "average_reward"

# 実行状態の値。文字列のまま持つ (Enum にするほど種類が多くないため)。
STATUS_FINISHED = "finished"      # ログ末尾に `time : ...` がある = 正常終了
STATUS_INCOMPLETE = "incomplete"  # 途中で止まっている (中断・クラッシュ・実行中)


@dataclass
class ExperimentRun:
    """1 回の学習実行 (train_log_XX.txt 1 本) に対応するレコード。"""

    run_id: str                    # 例: "Start202609301533/202609301533/01"
    batch_id: str                  # 例: "Start202609301533"  (実験スクリプト 1 回分)
    condition_id: str              # 例: "202609301533"       (config 1 つ分のディレクトリ)
    run_index: int | None          # train_log_XX.txt の XX
    algorithm: str | None          # train_args.agent.type (例: "SRS_v2_v3")
    variant: str | None            # 今は algorithm と同じ。将来 subtype 等を入れる余地
    seed: int | None               # train_args.seed (SRS_v2 では全て 0 で、run 間は区別できない)
    started_at: datetime | None    # ディレクトリ名 YYYYMMDDHHMM から復元
    status: str                    # STATUS_FINISHED / STATUS_INCOMPLETE
    epochs: int | None             # 記録されたエポック数
    episodes: int | None           # 最終的に生成されたエピソード数
    duration_sec: float | None     # `time : ...` の値 (秒)
    env: str | None                # env_args.env
    parameters: dict[str, Any]     # flatten した設定。例 {"train_args.metadata.global_eta": 0.1}
    metrics: dict[str, list[float | None]]  # エポック順の系列。例 {"win_rate": [...], "loss.total": [...]}
    source_path: str               # train_log の絶対パス
    config_path: str | None = None # 同じディレクトリの config_XX.yaml
    git_commit: str | None = None  # バックアップに記録が無いので常に None (欠損)
    purpose: str | None = None     # 同上 (欠損)
    memo: str | None = None        # 同上 (欠損)
    hints: list[str] = field(default_factory=list)  # バッチ内の PNG 名など、目的のヒントになる物
    warnings: list[str] = field(default_factory=list)  # 読み込めたが気になった点

    def final_value(self, metric: str = DEFAULT_SCORE_METRIC) -> float | None:
        """指定した指標の最終エポックの値。系列が無い/空なら None。"""
        series = self.metrics.get(metric)
        if not series:
            return None
        # 末尾から最初の非 None を探す (最終エポックの行が欠けていることがある)
        for value in reversed(series):
            if value is not None:
                return value
        return None

    @property
    def final_score(self) -> float | None:
        """既定指標 (DEFAULT_SCORE_METRIC) の最終値。"""
        return self.final_value(DEFAULT_SCORE_METRIC)


@dataclass
class LoadError:
    """読み込みに失敗したファイルとその理由。アプリを止めずに一覧で見せる。"""

    path: str
    reason: str


@dataclass
class SkippedItem:
    """エラーではないが読み込まなかった物 (未対応形式など)。"""

    path: str
    reason: str


@dataclass
class Attachment:
    """バッチに付随する図 (手作業で作った比較グラフなど)。ランではなくバッチ単位の情報。

    実験の目的や考察は図のファイル名にしか残っていないことが多いので、
    一覧・比較の近くで見られるようにする。ファイルは読むだけで、コピーや移動はしない。
    """

    batch_id: str      # 例: "Start202609301533"。trainlog 直下の図は root ディレクトリ名
    path: str          # 絶対パス
    name: str          # ファイル名 (表示用)
    condition_id: str | None = None  # 条件フォルダ内にあった図ならその条件 ID


@dataclass
class LoadResult:
    """ローダー全体の結果。runs と errors を分けて返すのがポイント。"""

    runs: list[ExperimentRun] = field(default_factory=list)
    errors: list[LoadError] = field(default_factory=list)
    skipped: list[SkippedItem] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)

    def extend(self, other: "LoadResult") -> None:
        self.runs.extend(other.runs)
        self.errors.extend(other.errors)
        self.skipped.extend(other.skipped)
        self.attachments.extend(other.attachments)

    def attachments_for(self, batch_ids: set[str] | None = None) -> list[Attachment]:
        """batch_ids に含まれるバッチの図 (None なら全部)。"""
        if batch_ids is None:
            return list(self.attachments)
        return [a for a in self.attachments if a.batch_id in batch_ids]
