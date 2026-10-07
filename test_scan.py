"""scan.py のテスト。tmp_path に最小の trainlog を作って確認する。  実行: pytest -q"""

from datetime import datetime
from pathlib import Path

from scan import differing_keys, scan, short

ARGS = "{'env_args': {'env': 'simple_pyramid'}, 'train_args': {'seed': 0, 'agent': {'type': 'SRS_v2_v3'}, 'metadata': {'global_eta': %s}}, 'worker_args': {'server_address': '0.0.0.0'}}"


def write_log(path: Path, eta: float = 0.1, finished: bool = True) -> None:
    body = ARGS % eta + "\nwaiting training\nepoch 0\nwin rate = 0.5 (1.0 / 2)\n"
    if finished:
        body += "finished server\ntime : 123.4\n"
    path.write_text(body)


def make_condition(cond_dir: Path, n_runs: int, eta: float = 0.1, unfinished: int = 0) -> None:
    cond_dir.mkdir(parents=True)
    (cond_dir / "config_01.yaml").write_text("dummy: 1\n")
    for i in range(1, n_runs + 1):
        write_log(cond_dir / f"train_log_{i:02d}.txt", eta, finished=i > unfinished)


def test_run_count_equals_number_of_train_logs(tmp_path: Path):
    make_condition(tmp_path / "Start202609301533" / "202609301533", n_runs=20)
    (tmp_path / "Start202609301533" / "eta_vs_c_floor.png").write_bytes(b"\x89PNG")

    batches, root_images = scan(tmp_path)
    assert len(batches) == 1 and root_images == []
    b = batches[0]
    assert b.name == "Start202609301533"
    assert b.started_at == datetime(2026, 9, 30, 15, 33)
    assert [p.name for p in b.images] == ["eta_vs_c_floor.png"]
    assert b.conditions[0].n_runs == 20          # train_log_20.txt まで -> 20 回平均
    assert b.conditions[0].n_finished == 20
    assert b.conditions[0].algorithm == "SRS_v2_v3"
    assert b.total_runs == 20


def test_unfinished_logs_are_counted_separately(tmp_path: Path):
    make_condition(tmp_path / "StartA" / "c1", n_runs=10, unfinished=3)
    (b,), _ = scan(tmp_path)
    assert b.conditions[0].n_runs == 10
    assert b.conditions[0].n_finished == 7


def test_multiple_conditions_and_differing_keys(tmp_path: Path):
    make_condition(tmp_path / "StartA" / "c1", n_runs=5, eta=0.1)
    make_condition(tmp_path / "StartA" / "c2", n_runs=5, eta=0.5)
    (b,), _ = scan(tmp_path)
    assert [c.name for c in b.conditions] == ["c1", "c2"]
    assert differing_keys(b.conditions) == ["train_args.metadata.global_eta"]  # seed や IP は出ない
    assert short("train_args.metadata.global_eta") == "global_eta"


def test_old_layout_and_root_images_and_order(tmp_path: Path):
    make_condition(tmp_path / "202606051724", n_runs=2)                 # 旧形式: 直下に条件フォルダ
    make_condition(tmp_path / "Start202609150119" / "202609150119", n_runs=3)
    (tmp_path / "202606051724" / "simple_pyramid.png").write_bytes(b"\x89PNG")
    (tmp_path / "comparison.PNG").write_bytes(b"\x89PNG")              # 拡張子は大文字でも可
    (tmp_path / "notes.txt").write_text("not an image")

    batches, root_images = scan(tmp_path)
    assert [b.name for b in batches] == ["Start202609150119", "202606051724"]  # 新しい順
    assert [p.name for p in root_images] == ["comparison.PNG"]
    old = batches[1]
    assert old.conditions[0].name == "202606051724" and old.conditions[0].n_runs == 2
    assert [p.name for p in old.images] == ["simple_pyramid.png"]


def test_broken_first_line_gives_empty_settings(tmp_path: Path):
    cond = tmp_path / "StartA" / "c1"
    cond.mkdir(parents=True)
    (cond / "train_log_01.txt").write_text("\x00garbage\n")
    (b,), _ = scan(tmp_path)
    assert b.conditions[0].n_runs == 1
    assert b.conditions[0].settings == {}
    assert b.conditions[0].algorithm == "不明"


def test_missing_or_empty_root(tmp_path: Path):
    assert scan(tmp_path / "nope") == ([], [])
    assert scan(tmp_path) == ([], [])
