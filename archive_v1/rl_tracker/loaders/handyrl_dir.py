"""HandyRL の trainlog ディレクトリ (展開済み) を読むローダー。

対応する構造 (bash_scripts/experiment_config.sh が生成する物):

    trainlog/
      Start202609301533/            <- バッチ (スクリプト 1 回の実行)
        202609301533/               <- 条件 (config 1 つ)
          config_01.yaml
          train_log_01.txt ... train_log_10.txt
        eta_vs_c_floor.png          <- 手作業の図。目的のヒントとして名前だけ拾う
      202606051724/                 <- 旧形式: root 直下に条件ディレクトリ
        config.yaml
        train_log_01.txt ...
      Start202609281636.zip         <- zip のみ。MVP では未対応として skipped に入れる

判定ルール: 「train_log_*.txt を含むディレクトリ」を条件ディレクトリとみなす。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from ..models import (
    STATUS_FINISHED,
    STATUS_INCOMPLETE,
    Attachment,
    ExperimentRun,
    LoadError,
    LoadResult,
    SkippedItem,
)
from ..parsers import (
    epochs_to_metrics,
    flatten,
    parse_config_yaml,
    parse_train_log,
    scalarize,
)

_RE_LOG_NAME = re.compile(r"^train_log_(\d+)\.txt$")
_RE_DIR_TIMESTAMP = re.compile(r"(\d{12})")  # YYYYMMDDHHMM
# 画面に表示できる画像形式。pdf は st.image で表示できないので対象外
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


class HandyRLDirLoader:
    name = "handyrl_dir"

    def can_handle(self, root: Path) -> bool:
        return root.is_dir()

    def load(self, root: Path) -> LoadResult:
        result = LoadResult()
        root = root.resolve()

        condition_dirs = sorted(
            {p.parent for p in root.rglob("train_log_*.txt")}
        )
        if not condition_dirs:
            result.errors.append(LoadError(str(root), "train_log_*.txt が 1 つも見つかりません"))
            return result

        for cond_dir in condition_dirs:
            result.extend(self._load_condition_dir(cond_dir, root))

        self._collect_zip_only_batches(root, condition_dirs, result)
        result.attachments = self._collect_images(root, condition_dirs)
        return result

    def _collect_images(self, root: Path, condition_dirs: list[Path]) -> list[Attachment]:
        """バッチ直下・条件ディレクトリ・trainlog 直下にある画像をパス付きで集める。

        画像は「この実験で何を見たかったか」を示す唯一の手がかりなので、
        ラン (train_log) とは別に、所属するバッチの ID を付けて保持する。
        """
        attachments: list[Attachment] = []
        seen: set[Path] = set()

        def add(directory: Path, batch_id: str, condition_id: str | None) -> None:
            if not directory.is_dir():
                return
            for p in sorted(directory.iterdir()):
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES and p not in seen:
                    seen.add(p)
                    attachments.append(Attachment(batch_id=batch_id, path=str(p), name=p.name, condition_id=condition_id))

        for cond_dir in condition_dirs:
            batch_dir = cond_dir.parent if cond_dir.parent != root else cond_dir
            if batch_dir != cond_dir:
                add(batch_dir, batch_dir.name, None)
            add(cond_dir, batch_dir.name, cond_dir.name)
        # trainlog 直下の図 (複数バッチをまたいだ比較図など) は root の名前でまとめる
        add(root, root.name, None)
        return attachments

    # ------------------------------------------------------------------
    def _load_condition_dir(self, cond_dir: Path, root: Path) -> LoadResult:
        result = LoadResult()

        # バッチ ID: root 直下に条件ディレクトリがある旧形式では条件 ID と同じにする
        batch_dir = cond_dir.parent if cond_dir.parent != root else cond_dir
        batch_id = batch_dir.name
        condition_id = cond_dir.name
        started_at = _timestamp_from_name(condition_id) or _timestamp_from_name(batch_id)

        # 目的のヒント: バッチ直下の画像ファイル名 (推測はせず、名前をそのまま持つだけ)
        hints = sorted(
            p.name for p in batch_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        ) if batch_dir != cond_dir else []

        # config.yaml は「ログ 1 行目が読めない時の代替」として使う
        config_path, config_params, config_warning = self._read_config(cond_dir)
        if config_warning:
            result.skipped.append(SkippedItem(str(cond_dir), config_warning))

        for log_path in sorted(cond_dir.glob("train_log_*.txt")):
            try:
                run = self._build_run(
                    log_path, batch_id, condition_id, started_at,
                    config_path, config_params, hints,
                )
                result.runs.append(run)
            except Exception as exc:  # noqa: BLE001 - 1 ファイルの失敗で全体を止めない
                result.errors.append(LoadError(str(log_path), f"{type(exc).__name__}: {exc}"))
        return result

    def _read_config(self, cond_dir: Path):
        """(config_path, flatten した params または None, 警告文字列または None)"""
        configs = sorted(cond_dir.glob("config*.yaml")) + sorted(cond_dir.glob("config*.yml"))
        if not configs:
            return None, None, None
        warning = None
        if len(configs) > 1:
            warning = (
                f"config が {len(configs)} 個あります ({', '.join(c.name for c in configs)})。"
                f" {configs[0].name} を代替用に使いますが、設定はログ 1 行目を優先します"
            )
        try:
            params = flatten(parse_config_yaml(configs[0].read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            return str(configs[0]), None, f"{configs[0].name} を読めません: {type(exc).__name__}: {exc}"
        return str(configs[0]), params, warning

    def _build_run(
        self, log_path: Path, batch_id: str, condition_id: str, started_at,
        config_path, config_params, hints,
    ) -> ExperimentRun:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise ValueError("ファイルが空です")

        parsed = parse_train_log(text)
        warnings = list(parsed.warnings)

        # 設定行もエポックも無いなら、これは train_log ではない (壊れている) と判断する。
        # 設定行だけで終わっているファイルは「開始直後に止まったラン」なので読み込む。
        if parsed.args is None and not parsed.epochs:
            raise ValueError("train_log として解釈できません (1行目の設定もエポック情報も無し)")

        # 設定: ログ 1 行目 (そのランの実際の設定) > config.yaml (同じ条件ディレクトリ) > なし
        if parsed.args is not None:
            params = flatten(parsed.args)
        elif config_params is not None:
            params = dict(config_params)
            warnings.append("設定は config.yaml から補いました")
        else:
            params = {}
            warnings.append("設定が見つかりません (ログ 1 行目も config.yaml も無し)")

        params = {k: scalarize(v) for k, v in params.items()}
        m = _RE_LOG_NAME.match(log_path.name)
        run_index = int(m.group(1)) if m else None
        metrics = epochs_to_metrics(parsed.epochs)

        # エピソード数はカウンタ行 ("50 100 150 ...") の最大値。無ければ推測せず None
        episodes = parsed.max_episode_count
        if episodes is None and parsed.epochs:
            warnings.append("エピソード数カウンタ行が無いため episodes は None")

        algorithm = params.get("train_args.agent.type")
        return ExperimentRun(
            run_id=f"{batch_id}/{condition_id}/{run_index:02d}" if run_index is not None else f"{batch_id}/{condition_id}/{log_path.stem}",
            batch_id=batch_id,
            condition_id=condition_id,
            run_index=run_index,
            algorithm=algorithm,
            variant=algorithm,
            seed=_as_int(params.get("train_args.seed")),
            started_at=started_at,
            status=STATUS_FINISHED if parsed.finished else STATUS_INCOMPLETE,
            epochs=len(parsed.epochs),  # 0 = 開始したがエポックに到達しなかった
            episodes=episodes,
            duration_sec=parsed.duration_sec,
            env=params.get("env_args.env"),
            parameters=params,
            metrics=metrics,
            source_path=str(log_path),
            config_path=config_path,
            hints=list(hints),
            warnings=warnings,
        )

    def _collect_zip_only_batches(self, root: Path, condition_dirs: list[Path], result: LoadResult) -> None:
        """展開済みディレクトリが無い zip を『未対応』として記録する (読み込みはしない)。"""
        extracted = {d.name for d in condition_dirs} | {d.parent.name for d in condition_dirs}
        for zip_path in sorted(root.glob("*.zip")):
            if zip_path.stem not in extracted:
                result.skipped.append(
                    SkippedItem(str(zip_path), "zip のみで展開済みディレクトリが無いため未読み込み (MVP では zip 未対応)")
                )


def _timestamp_from_name(name: str) -> datetime | None:
    """'Start202609301533' / '202609301533' -> datetime(2026, 9, 30, 15, 33)。合わなければ None。"""
    m = _RE_DIR_TIMESTAMP.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M")
    except ValueError:
        return None


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
