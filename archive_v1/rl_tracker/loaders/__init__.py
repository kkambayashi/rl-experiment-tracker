"""ローダーの登録と入口関数。

形式を増やすときは LOADERS にインスタンスを追加する。
"""

from __future__ import annotations

from pathlib import Path

from ..models import LoadError, LoadResult
from .base import Loader
from .handyrl_dir import HandyRLDirLoader

LOADERS: list[Loader] = [
    HandyRLDirLoader(),
]


def load_all(root: str | Path) -> LoadResult:
    """root を扱えるローダーを順に適用して結果をまとめる。

    root が存在しない等の致命的な問題も、例外ではなく LoadResult.errors として返す
    (UI 側で「理由付きで表示」するため)。
    """
    path = Path(root).expanduser()
    result = LoadResult()

    if not path.exists():
        result.errors.append(LoadError(str(path), "パスが存在しません"))
        return result

    handled = False
    for loader in LOADERS:
        try:
            if not loader.can_handle(path):
                continue
            handled = True
            result.extend(loader.load(path))
        except Exception as exc:  # noqa: BLE001 - ローダーのバグでもアプリは落とさない
            result.errors.append(LoadError(str(path), f"{loader.name}: {type(exc).__name__}: {exc}"))

    if not handled:
        result.errors.append(LoadError(str(path), "対応するローダーがありません (ディレクトリを指定してください)"))
    return result


__all__ = ["LOADERS", "Loader", "HandyRLDirLoader", "load_all"]
