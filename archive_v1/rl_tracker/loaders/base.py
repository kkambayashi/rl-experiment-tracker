"""ローダーの共通インターフェース。

新しい保存形式に対応するときは、この Protocol を満たすクラスを loaders/ に追加し、
loaders/__init__.py の LOADERS に登録するだけでよい (他のモジュールは変更不要)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import LoadResult


class Loader(Protocol):
    name: str

    def can_handle(self, root: Path) -> bool:
        """この root をこのローダーが扱えるか (安価な判定だけ行う)。"""
        ...

    def load(self, root: Path) -> LoadResult:
        """root 以下を走査して LoadResult を返す。例外は内部で捕まえて errors に入れる。"""
        ...
