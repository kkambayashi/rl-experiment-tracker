"""サンプルデータ生成スクリプト。

実際の SRS_v2 バックアップと同じディレクトリ構造・ログ形式の「合成データ」を
sample_data/trainlog/ に作る。研究データそのものは含まない (数値は乱数で生成)。

含めている状況:
  - 2 つのバッチにまたがる同一条件 (SRS_v2_v3, eta=0.1) -> 実行回数の合算を確認できる
  - 未完了ラン (time 行なし)                       -> status=incomplete
  - 設定行だけで止まったラン (epoch 0 件)          -> status=incomplete, epochs=0
  - 壊れたログ / 空のログ                          -> 読み込みエラー一覧に出る
  - 壊れた config.yaml                             -> 警告 (ログ 1 行目から設定は取れる)
  - 旧形式 (root 直下の条件ディレクトリ + config.yaml)
  - zip のみのバッチ                               -> 未対応として表示

使い方:  python scripts/make_sample_data.py [出力先 (既定: sample_data/trainlog)]
"""

from __future__ import annotations

import copy
import io
import random
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "sample_data" / "trainlog"

EPOCHS = 30
EPISODES_PER_EPOCH = 300
WARMUP_EPISODES = 10000

# HandyRL の config.yaml と同じ構造の最小セット。server_address はダミー。
BASE_ARGS = {
    "env_args": {
        "env": "simple_pyramid",
        "param": {
            "depth": 6,
            "hyperplane_dim": 2,
            "general_seed": -1,
            "start_random": True,
            "features": {"seed": -1, "dim": 2, "obs_var": 0.01},
            "rewards": {"depth": [6], "coordinates": [[1, 5]], "type": ["fix"], "mu": [1.0], "var": [0.0]},
        },
    },
    "train_args": {
        "default_learning_rate": 3e-08,
        "gamma": 1.0,
        "forward_steps": 2,
        "entropy_regularization": 0.001,
        "update_episodes": EPISODES_PER_EPOCH,
        "batch_size": 256,
        "minimum_episodes": WARMUP_EPISODES,
        "maximum_episodes": 20000,
        "epochs": EPOCHS,
        "eval_rate": 0.1,
        "policy_target": "TD-Q",
        "value_target": "TD-Q",
        "eval": {"opponent": ["random"]},
        "seed": 0,
        "agent": {"type": "SRS_v2_v3", "use_RND": False},
        "metadata": {
            "name": ["knn", "global_aleph", "regional_weight", "global_return_size", "global_eta"],
            "eval_base": "QL",
            "knn": {"size": 10000, "k": 64},
            "regional_weight": 1.0,
            "global_aleph": 1.0,
            "global_return_method": "Mean",
            "global_return_size": 100,
            "global_eta": 0.1,
        },
    },
    "worker_args": {"server_address": "0.0.0.0", "num_parallel": 128},
}


def make_args(agent_type: str, **overrides) -> dict:
    args = copy.deepcopy(BASE_ARGS)
    args["train_args"]["agent"]["type"] = agent_type
    md = args["train_args"]["metadata"]
    if agent_type == "BASE":
        args["train_args"]["policy_target"] = "TD"
        args["train_args"]["value_target"] = "TD"
        args["train_args"]["metadata"] = {"name": []}
    for k, v in overrides.items():
        md[k] = v
        if k not in md["name"]:
            md["name"].append(k)
    return args


def loss_keys(agent_type: str) -> list[str]:
    if agent_type == "BASE":
        return ["p", "v", "ent", "total"]
    if agent_type == "RSRS":
        return ["p", "v", "q", "rnd", "entropy_srs", "ent_c_reg", "ent", "total"]
    return ["p", "v", "q", "entropy_srs", "ent_c_reg", "ent", "total"]


def render_log(args: dict, rng: random.Random, speed: float, n_epochs: int = EPOCHS, finished: bool = True) -> str:
    """HandyRL の標準出力を模したログ文字列を作る。speed が大きいほど早く勝率が上がる。"""
    out = io.StringIO()
    out.write(repr(args) + "\n")
    out.write("waiting training\nstarted batcher 0\nstarted server\nstarted gather 0\nopened worker 0\n")
    counter = " ".join(str(i) for i in range(50, WARMUP_EPISODES + 1, 50))
    out.write(f"{counter} start sender\nstart receiver\nstarted training\n")

    episodes = WARMUP_EPISODES
    keys = loss_keys(args["train_args"]["agent"]["type"])
    for epoch in range(n_epochs):
        nums = [episodes + 50 * (i + 1) for i in range(EPISODES_PER_EPOCH // 50)]
        episodes = nums[-1]
        out.write(" ".join(str(n) for n in nums) + " \n")
        out.write(f"\nepoch {epoch}\n")

        progress = 1 - (1 - 0.0) * (2.718 ** (-speed * epoch / n_epochs * 4))
        if epoch == 0:
            win, games = 0.5, 4378
        else:
            games = rng.choice([127, 128])
            win = min(1.0, max(0.0, 0.5 + 0.5 * progress + rng.gauss(0, 0.03)))
        wins = round(win * games, 1)
        out.write(f"win rate = {win:.3f} ({wins} / {games})\n")
        out.write(f"average reward = {2 * win - 1:.3f} ({wins} / {games})\n")
        gen_mean = min(1.0, max(0.0, progress * 0.95 + rng.gauss(0, 0.05)))
        gen_std = max(0.0, (1 - gen_mean) * 0.4 + rng.gauss(0, 0.02))
        out.write(f"generation stats = {gen_mean:.3f} +- {gen_std:.3f}\n")
        losses = {k: max(0.0, (1 - progress) * rng.uniform(0.01, 0.2)) for k in keys}
        losses["ent"] = 1.38 * (1 - 0.3 * progress)
        losses["total"] = sum(v for k, v in losses.items() if k not in ("ent", "total"))
        out.write("loss = " + " ".join(f"{k}:{v:.3f}" for k, v in losses.items()) + "\n")
        out.write(f"updated model({(epoch + 1) * 87})\n")

    if finished:
        out.write("closed worker 0\ndisconnected\ndisconnected\nfinished server\n")
        out.write(f"time : {rng.uniform(500, 3300):.6f}\n")
    return out.getvalue()


def make_plot_png(rng: random.Random, n_lines: int = 2, width: int = 480, height: int = 300) -> bytes:
    """学習曲線風の簡易画像を、外部ライブラリなし (zlib + struct) で PNG にする。

    実験バックアップに入っている手作業の比較図 (matplotlib の png) の代わり。
    中身は乱数の折れ線なので意味は無いが、「図」タブの動作確認に使える。
    """
    import struct
    import zlib

    white, axis = (255, 255, 255), (60, 60, 60)
    colors = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0)]
    pixels = [[white] * width for _ in range(height)]
    left, right, top, bottom = 50, width - 20, 20, height - 40

    def put(x: int, y: int, c: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            pixels[y][x] = c

    for x in range(left, right):           # x 軸
        put(x, bottom, axis)
    for y in range(top, bottom):           # y 軸
        put(left, y, axis)
    for i in range(1, 5):                  # 目盛り (薄いグリッド)
        gy = bottom - (bottom - top) * i // 5
        for x in range(left, right, 3):
            put(x, gy, (220, 220, 220))

    for k in range(n_lines):
        color = colors[k % len(colors)]
        speed = rng.uniform(0.6, 1.6)
        prev = None
        for x in range(left + 1, right):
            t = (x - left) / (right - left)
            value = 1 - 2.718 ** (-speed * t * 4) + rng.gauss(0, 0.03)
            y = int(bottom - (bottom - top) * min(1.0, max(0.0, value)))
            if prev is not None:           # 前の点と縦に結んで折れ線にする
                lo, hi = sorted((prev, y))
                for yy in range(lo, hi + 1):
                    put(x, yy, color)
                    put(x, yy + 1, color)
            prev = y

    raw = b"".join(b"\x00" + bytes(ch for px in row for ch in px) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit RGB
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def write_condition(cond_dir: Path, args: dict, config_name: str, n_runs: int, rng: random.Random,
                    speed: float, broken_runs: dict[int, str] | None = None) -> None:
    cond_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / config_name).write_text(yaml.safe_dump(args, sort_keys=False, allow_unicode=True), encoding="utf-8")
    for i in range(1, n_runs + 1):
        path = cond_dir / f"train_log_{i:02d}.txt"
        kind = (broken_runs or {}).get(i)
        if kind == "incomplete":
            path.write_text(render_log(args, rng, speed, n_epochs=12, finished=False), encoding="utf-8")
        elif kind == "args_only":
            path.write_text(repr(args) + "\n", encoding="utf-8")
        elif kind == "garbage":
            path.write_bytes(bytes(rng.getrandbits(8) for _ in range(300)))
        elif kind == "empty":
            path.write_text("", encoding="utf-8")
        else:
            path.write_text(render_log(args, rng, speed), encoding="utf-8")


def main(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rng = random.Random(20260930)

    # --- バッチ 1: アルゴリズム / eta の比較 (5 run ずつ) -----------------------
    b1 = out / "Start202609150119"
    write_condition(b1 / "202609150119", make_args("BASE"), "config_01.yaml", 5, rng, speed=0.6)
    write_condition(b1 / "202609150300", make_args("SRS_v2_v3", global_eta=0.1), "config_02.yaml", 5, rng, speed=1.4)
    write_condition(b1 / "202609150450", make_args("SRS_v2_v3", global_eta=0.5), "config_03.yaml", 5, rng, speed=1.0)
    (b1 / "SRS_v2_v3実装η比較.png").write_bytes(make_plot_png(rng, n_lines=3))

    # --- バッチ 2: eta=0.1 の追加実行 + c_floor 導入 (未完了・壊れたログを含む) ---
    b2 = out / "Start202609221614"
    write_condition(b2 / "202609221614", make_args("SRS_v2_v3", global_eta=0.1), "config_01.yaml", 3, rng, speed=1.3)
    write_condition(
        b2 / "202609221800", make_args("SRS_v2_c_floor", global_eta=0.1, c_floor=0.001), "config_02.yaml", 6, rng,
        speed=1.5, broken_runs={3: "incomplete", 4: "args_only", 5: "garbage", 6: "empty"},
    )
    # 壊れた config.yaml (ログ 1 行目から設定は取れるので実験自体は読み込める)
    (b2 / "202609221800" / "config_02.yaml").write_text("train_args: [unclosed\n  - : :\n", encoding="utf-8")
    (b2 / "eta_vs_c_floor.png").write_bytes(make_plot_png(rng, n_lines=2))

    # --- 旧形式: root 直下に条件ディレクトリ --------------------------------
    old = out / "202606051724"
    write_condition(old, make_args("RSRS"), "config.yaml", 2, rng, speed=0.9)
    (old / "retruns.csv").write_text(",".join(f"{rng.random():.5f}" for _ in range(20)) + "\n", encoding="utf-8")
    (old / "simple_pyramid.png").write_bytes(make_plot_png(rng, n_lines=1))

    # trainlog 直下に置かれた、複数バッチをまたぐ比較図
    (out / "SRS_v2_v3_eta_0.1-1.0_comparison.png").write_bytes(make_plot_png(rng, n_lines=4))

    # --- zip のみのバッチ (展開済みディレクトリなし) ------------------------
    zpath = out / "Start202609281636.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zargs = make_args("SRS_v2_v3", global_eta=0.2)
        zf.writestr("Start202609281636/202609281636/config_01.yaml", yaml.safe_dump(zargs, sort_keys=False))
        zf.writestr("Start202609281636/202609281636/train_log_01.txt", repr(zargs) + "\n")

    # 展開済みがあるバッチの zip は重複なので黙って無視されることの確認用
    with zipfile.ZipFile(out / "Start202609150119.zip", "w") as zf:
        zf.writestr("Start202609150119/.gitkeep", "")

    n_logs = len(list(out.rglob("train_log_*.txt")))
    print(f"wrote {n_logs} logs under {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
