"""trainlog フォルダを走査して、バッチ (Start... フォルダ) ごとの「図」と「実行回数」をまとめる。

対象の構造 (bash_scripts/experiment_config.sh が作る物):

    trainlog/
      Start202609301533/            <- バッチ
        202609301533/               <- 条件 (config 1 つ分)
          config_01.yaml
          train_log_01.txt ... train_log_20.txt   <- 本数 = その条件を回した回数 (= 平均の回数)
        eta_vs_c_floor.png          <- 手作業で保存した図
      202606051724/                 <- 旧形式: trainlog 直下に条件フォルダ
        train_log_01.txt ...

ファイルは読むだけで、書き込み・移動はしない。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_RE_TIMESTAMP = re.compile(r"(\d{12})")  # フォルダ名の YYYYMMDDHHMM

# 条件の違いとして表示しないキー (サーバーの IP など、実験条件ではない物)
IGNORED_KEYS = {"worker_args.server_address", "worker_args.num_parallel", "train_args.metadata.name"}


@dataclass
class Condition:
    name: str                 # 条件フォルダ名 (例: "202609301533")
    n_runs: int               # train_log_XX.txt の本数 = 平均に使った回数
    n_finished: int           # 末尾に "time :" がある本数 (正常終了したラン)
    settings: dict            # train_log 1 行目の設定を平坦化した物。読めなければ {}

    @property
    def algorithm(self) -> str:
        return str(self.settings.get("train_args.agent.type", "不明"))


@dataclass
class Batch:
    name: str                 # 例: "Start202609301533"
    path: Path
    started_at: datetime | None
    images: list[Path] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)

    @property
    def total_runs(self) -> int:
        return sum(c.n_runs for c in self.conditions)


def scan(root: str | Path) -> tuple[list[Batch], list[Path]]:
    """root 以下を走査し、(バッチの一覧 [新しい順], trainlog 直下の画像) を返す。

    root が無い・train_log が 1 つも無い場合は ([], []) を返す (例外にしない)。
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        return [], []

    # 「train_log_*.txt を含むフォルダ」を条件フォルダとみなす
    condition_dirs = sorted({p.parent for p in root.rglob("train_log_*.txt")})

    batches: dict[Path, Batch] = {}
    for cond_dir in condition_dirs:
        # 旧形式 (root 直下に条件フォルダ) では、条件フォルダ自身をバッチとして扱う
        batch_dir = cond_dir.parent if cond_dir.parent != root else cond_dir
        if batch_dir not in batches:
            batches[batch_dir] = Batch(
                name=batch_dir.name,
                path=batch_dir,
                started_at=_timestamp(batch_dir.name),
                images=_images_in(batch_dir),
            )
        batch = batches[batch_dir]
        batch.conditions.append(_read_condition(cond_dir))
        if cond_dir != batch_dir:
            batch.images += [p for p in _images_in(cond_dir) if p not in batch.images]

    ordered = sorted(batches.values(), key=lambda b: b.name, reverse=True)  # 名前 = 日時なので新しい順
    return ordered, _images_in(root)


def _read_condition(cond_dir: Path) -> Condition:
    logs = sorted(cond_dir.glob("train_log_*.txt"))
    return Condition(
        name=cond_dir.name,
        n_runs=len(logs),
        n_finished=sum(1 for p in logs if _is_finished(p)),
        settings=_read_settings(logs[0]) if logs else {},
    )


def _is_finished(log_path: Path) -> bool:
    """HandyRL のログは正常終了すると最後に 'time : 秒数' を出力する。末尾だけ読んで判定する。"""
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)                       # ファイル末尾へ
            f.seek(max(0, f.tell() - 300))     # 最後の 300 バイトだけ読む (巨大ログでも速い)
            return b"time :" in f.read()
    except OSError:
        return False


def _read_settings(log_path: Path) -> dict:
    """train_log の 1 行目は実行時の設定 (Python の dict 表現)。ast.literal_eval で安全に復元する。"""
    try:
        with log_path.open(encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        value = ast.literal_eval(first) if first.startswith("{") else None
    except (ValueError, SyntaxError, OSError):
        return {}
    return _flatten(value) if isinstance(value, dict) else {}


def _flatten(d: dict, prefix: str = "") -> dict:
    """{'a': {'b': 1}} -> {'a.b': 1}。list はそのまま値として残す。"""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _timestamp(name: str) -> datetime | None:
    m = _RE_TIMESTAMP.search(name)
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M") if m else None
    except ValueError:
        return None


# --------------------------------------------------------------------------
# 表示用
# --------------------------------------------------------------------------
def differing_keys(conditions: list[Condition]) -> list[str]:
    """バッチ内の条件同士で値が違う設定キー (= その図が比較している物)。"""
    keys: set[str] = set()
    for c in conditions:
        keys.update(c.settings)
    diff = []
    for key in sorted(keys - IGNORED_KEYS):
        values = {repr(c.settings.get(key, "—")) for c in conditions}
        if len(values) > 1:
            diff.append(key)
    return diff


def short(key: str) -> str:
    """'train_args.metadata.global_eta' -> 'global_eta' のように、決まった前置きを外して短くする。"""
    for prefix in ("train_args.metadata.", "train_args.agent.", "train_args.", "env_args.param.", "env_args."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key
