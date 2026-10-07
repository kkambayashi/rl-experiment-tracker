"""HandyRL 形式の train_log / config.yaml を解析する純粋関数。

ファイル I/O はここでは行わない (文字列を受け取り、構造を返す)。
そのため pytest で文字列を渡すだけでテストできる。

train_log_XX.txt の構造 (research code: handyrl/train.py の print 文に対応):

    {'env_args': {...}, 'train_args': {...}, ...}   <- 1 行目: 全設定の Python dict
    waiting training
    50 100 150 ... 10000 start sender                <- エピソード数カウンタ (空白区切り)
    epoch 0
    win rate = 0.500 (2189.0 / 4378)
    average reward = 0.000 (2189.0 / 4378)
    generation stats = 0.071 +- 0.257
    loss = p:0.126 v:0.042 q:0.039 total:0.206      <- キーはアルゴリズムで増減
    updated model(89)
    ...
    finished server
    time : 3096.573837242089                         <- これが無ければ未完了
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

# --- 正規表現 ---------------------------------------------------------------
# 行頭アンカー付き。数値は "Nan" のこともある (評価が 0 件のエポック)。
_RE_EPOCH = re.compile(r"^epoch (\d+)\s*$")
_RE_WIN_RATE = re.compile(r"^win rate(?: \((.+?)\))? = (\S+) \(([\d.]+) / (\d+)\)")
_RE_AVG_REWARD = re.compile(r"^average reward(?: \((.+?)\))? = (\S+) \(([\d.]+) / (\d+)\)")
_RE_GEN_STATS = re.compile(r"^generation stats = (\S+) \+- (\S+)")
_RE_GEN_NAN = re.compile(r"^generation stats = Nan")
_RE_LOSS = re.compile(r"^loss = (.*)$")
_RE_UPDATED = re.compile(r"^updated model\((\d+)\)")
_RE_TIME = re.compile(r"^time :\s*([\d.eE+-]+)")
_RE_COUNTER_LINE = re.compile(r"^(\d+\s+)+(\d+\s*)?(start sender)?\s*$")


@dataclass
class EpochRecord:
    """1 エポック分の評価結果。無い項目は None。"""

    epoch: int
    win_rate: float | None = None
    average_reward: float | None = None
    eval_games: int | None = None          # win rate の分母 (評価対局数)
    generation_mean: float | None = None   # 学習中の軌跡のリターン平均
    generation_std: float | None = None
    losses: dict[str, float] = field(default_factory=dict)
    model_steps: int | None = None         # updated model(N) の N
    # 対戦相手が複数ある場合の個別値。例 {"random": 0.6}
    win_rate_by_opponent: dict[str, float] = field(default_factory=dict)


@dataclass
class ParsedLog:
    args: dict[str, Any] | None            # 1 行目の設定 dict。解釈できなければ None
    epochs: list[EpochRecord]
    finished: bool                         # `time :` 行があったか
    duration_sec: float | None
    max_episode_count: int | None          # カウンタ行で見えた最大エピソード数
    warnings: list[str] = field(default_factory=list)


def _to_float(token: str) -> float | None:
    """'0.656' -> 0.656, 'Nan' -> None。それ以外の文字列も None にする。"""
    if token.lower() == "nan":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_args_line(line: str) -> dict[str, Any] | None:
    """ログ 1 行目の Python dict 表現を dict に戻す。

    ast.literal_eval は eval と違い、リテラル (dict/list/数値/文字列/True/False/None) しか
    評価しないので、ログに悪意あるコードが混ざっていても実行されない。
    """
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        value = ast.literal_eval(line)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def parse_train_log(text: str) -> ParsedLog:
    """train_log_XX.txt の全文を ParsedLog に変換する。

    壊れた行は読み飛ばして warnings に記録し、例外は投げない
    (呼び出し側で「読めた範囲の情報」として扱えるようにするため)。
    """
    lines = text.splitlines()
    warnings: list[str] = []

    args = parse_args_line(lines[0]) if lines else None
    if args is None:
        warnings.append("1行目の設定 dict を解釈できませんでした (config.yaml があればそちらを使います)")

    epochs: list[EpochRecord] = []
    current: EpochRecord | None = None
    finished = False
    duration: float | None = None
    max_episodes: int | None = None

    for lineno, raw in enumerate(lines[1:], start=2):
        line = raw.rstrip()
        if not line:
            continue

        m = _RE_EPOCH.match(line)
        if m:
            current = EpochRecord(epoch=int(m.group(1)))
            epochs.append(current)
            continue

        m = _RE_TIME.match(line)
        if m:
            finished = True
            duration = _to_float(m.group(1))
            continue

        # エポック数カウンタ行: "50 100 150 ... 10000 start sender" / "10350 10400 ..."
        if _RE_COUNTER_LINE.match(line):
            nums = [int(t) for t in line.split() if t.isdigit()]
            if nums:
                max_episodes = max(max_episodes or 0, max(nums))
            continue

        if current is None:
            # "waiting training" "started server" などの状態行。情報としては不要なので無視
            continue

        m = _RE_WIN_RATE.match(line)
        if m:
            tag, value, _, games = m.groups()
            v = _to_float(value)
            if tag is None or tag == "total":
                current.win_rate = v
                current.eval_games = int(games)
            elif v is not None:
                current.win_rate_by_opponent[tag] = v
            continue

        m = _RE_AVG_REWARD.match(line)
        if m:
            tag, value, _, _ = m.groups()
            if tag is None or tag == "total":
                current.average_reward = _to_float(value)
            continue

        if _RE_GEN_NAN.match(line):
            continue  # 値なし → None のまま

        m = _RE_GEN_STATS.match(line)
        if m:
            current.generation_mean = _to_float(m.group(1))
            current.generation_std = _to_float(m.group(2))
            continue

        m = _RE_LOSS.match(line)
        if m:
            current.losses = _parse_loss_tokens(m.group(1), warnings, lineno)
            continue

        m = _RE_UPDATED.match(line)
        if m:
            current.model_steps = int(m.group(1))
            continue

        if line.startswith("win rate = Nan"):
            continue  # 評価 0 件のエポック。None のまま

        # ここに来る行は想定外。全部は出さず、最初の数件だけ記録する
        if len(warnings) < 5 and not _is_known_noise(line):
            warnings.append(f"{lineno}行目を解釈できませんでした: {line[:60]!r}")

    return ParsedLog(
        args=args,
        epochs=epochs,
        finished=finished,
        duration_sec=duration,
        max_episode_count=max_episodes,
        warnings=warnings,
    )


_KNOWN_NOISE_PREFIXES = (
    "waiting training", "started", "opened worker", "closed worker",
    "start sender", "start receiver", "finished", "disconnected", "connected",
)


def _is_known_noise(line: str) -> bool:
    return line.startswith(_KNOWN_NOISE_PREFIXES)


def _parse_loss_tokens(body: str, warnings: list[str], lineno: int) -> dict[str, float]:
    """'p:0.126 v:0.042 total:0.206' -> {'p': 0.126, 'v': 0.042, 'total': 0.206}"""
    losses: dict[str, float] = {}
    for token in body.split():
        if ":" not in token:
            continue
        key, _, value = token.rpartition(":")
        v = _to_float(value)
        if v is None:
            if len(warnings) < 5:
                warnings.append(f"{lineno}行目の loss 値を数値にできません: {token!r}")
            continue
        losses[key] = v
    return losses


# --- config.yaml --------------------------------------------------------------

def parse_config_yaml(text: str) -> dict[str, Any]:
    """config.yaml を dict にする。壊れていれば yaml.YAMLError がそのまま上がる。

    safe_load を使うので任意オブジェクトの生成は起きない。
    """
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("YAML のトップレベルが dict ではありません")
    return data


# --- 共通ユーティリティ -------------------------------------------------------

def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """ネストした dict を 'a.b.c' 形式のキーに平坦化する。

    list はそのまま値として残す (後で表示・比較しやすいよう repr 文字列化は呼び出し側で判断)。
    例: {'train_args': {'metadata': {'global_eta': 0.1}}}
        -> {'train_args.metadata.global_eta': 0.1}
    """
    out: dict[str, Any] = {}
    for key, value in d.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, full))
        else:
            out[full] = value
    return out


def scalarize(value: Any) -> Any:
    """list/tuple/dict を文字列化して、pandas の列やグループ化キーとして扱えるようにする。"""
    if isinstance(value, (list, tuple, dict)):
        return repr(value)
    return value


def epochs_to_metrics(epochs: list[EpochRecord]) -> dict[str, list[float | None]]:
    """EpochRecord の列を『指標名 -> エポック順の値リスト』に組み替える。

    loss のキーはログによって違うので、出現した全キーを集めてから整列する。
    """
    loss_keys: list[str] = []
    for e in epochs:
        for k in e.losses:
            if k not in loss_keys:
                loss_keys.append(k)

    metrics: dict[str, list[float | None]] = {
        "win_rate": [e.win_rate for e in epochs],
        "average_reward": [e.average_reward for e in epochs],
        "generation_mean": [e.generation_mean for e in epochs],
        "generation_std": [e.generation_std for e in epochs],
    }
    for k in loss_keys:
        metrics[f"loss.{k}"] = [e.losses.get(k) for e in epochs]
    return metrics
