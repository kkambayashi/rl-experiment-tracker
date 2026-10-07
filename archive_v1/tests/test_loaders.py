"""loaders のテスト。tmp_path に最小のディレクトリ構造を作って読み込む。"""

from datetime import datetime
from pathlib import Path

import pytest

from rl_tracker.loaders import load_all
from rl_tracker.models import STATUS_FINISHED, STATUS_INCOMPLETE

ARGS = (
    "{'env_args': {'env': 'simple_pyramid'}, 'train_args': {'seed': 0, "
    "'agent': {'type': 'SRS_v2_v3'}, 'metadata': {'global_eta': 0.1}}, "
    "'worker_args': {'server_address': '0.0.0.0'}}"
)


def _log(epochs: int, finished: bool = True, args: str = ARGS) -> str:
    lines = [args, "waiting training", "50 100 150 start sender"]
    for i in range(epochs):
        lines += [
            f"{200 + 50 * i} {250 + 50 * i} ",
            f"epoch {i}",
            f"win rate = {0.5 + 0.1 * i:.3f} (10.0 / 20)",
            f"average reward = {0.1 * i:.3f} (10.0 / 20)",
            f"generation stats = {0.2 * i:.3f} +- 0.100",
            "loss = p:0.1 v:0.2 total:0.3",
            f"updated model({100 * (i + 1)})",
        ]
    if finished:
        lines += ["finished server", "time : 123.5"]
    return "\n".join(lines) + "\n"


CONFIG = "train_args:\n  seed: 0\n  agent:\n    type: 'SRS_v2_v3'\n  metadata:\n    global_eta: 0.1\n"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "trainlog"
    cond = root / "Start202609301533" / "202609301533"
    cond.mkdir(parents=True)
    (cond / "config_01.yaml").write_text(CONFIG)
    (cond / "train_log_01.txt").write_text(_log(3))
    (cond / "train_log_02.txt").write_text(_log(2, finished=False))
    (cond / "train_log_03.txt").write_text(ARGS + "\n")              # 開始直後に停止
    (cond / "train_log_04.txt").write_text("\x00garbage\n")           # 壊れたファイル
    (cond / "train_log_05.txt").write_text("")                        # 空
    (root / "Start202609301533" / "eta_vs_c_floor.png").write_bytes(b"\x89PNG")
    (root / "Start202609281636.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # zip のみ
    return root


def test_load_all_separates_runs_and_errors(tree: Path):
    result = load_all(tree)
    assert sorted(r.run_index for r in result.runs) == [1, 2, 3]
    assert len(result.errors) == 2
    reasons = {Path(e.path).name: e.reason for e in result.errors}
    assert "train_log_04.txt" in reasons and "train_log_05.txt" in reasons
    assert "空" in reasons["train_log_05.txt"]


def test_run_fields_from_directory_layout(tree: Path):
    run = next(r for r in load_all(tree).runs if r.run_index == 1)
    assert run.run_id == "Start202609301533/202609301533/01"
    assert run.batch_id == "Start202609301533"
    assert run.condition_id == "202609301533"
    assert run.started_at == datetime(2026, 9, 30, 15, 33)
    assert run.algorithm == "SRS_v2_v3"
    assert run.seed == 0
    assert run.env == "simple_pyramid"
    assert run.parameters["train_args.metadata.global_eta"] == 0.1
    assert run.status == STATUS_FINISHED
    assert run.epochs == 3
    assert run.episodes == 350
    assert run.duration_sec == 123.5
    assert run.final_score == pytest.approx(0.2)          # average_reward の最終値
    assert run.final_value("win_rate") == pytest.approx(0.7)
    assert run.hints == ["eta_vs_c_floor.png"]
    assert run.config_path.endswith("config_01.yaml")
    # 記録が無い物は None のまま
    assert run.git_commit is None and run.purpose is None and run.memo is None


def test_incomplete_and_args_only_runs(tree: Path):
    runs = {r.run_index: r for r in load_all(tree).runs}
    assert runs[2].status == STATUS_INCOMPLETE and runs[2].epochs == 2 and runs[2].duration_sec is None
    assert runs[3].status == STATUS_INCOMPLETE and runs[3].epochs == 0
    assert runs[3].final_score is None
    assert runs[3].algorithm == "SRS_v2_v3"  # 設定行は読めている


def test_attachments_collect_images_with_batch_id(tree: Path):
    batch = tree / "Start202609301533"
    (batch / "memo.txt").write_text("not an image")                          # 画像以外は対象外
    (batch / "figure.PDF").write_bytes(b"%PDF")                                # pdf は表示できないので対象外
    (batch / "202609301533" / "env.jpg").write_bytes(b"\xff\xd8")            # 条件フォルダ内の画像
    (tree / "all_batches_comparison.png").write_bytes(b"\x89PNG")            # trainlog 直下の画像

    result = load_all(tree)
    by_name = {a.name: a for a in result.attachments}
    assert set(by_name) == {"eta_vs_c_floor.png", "env.jpg", "all_batches_comparison.png"}
    assert by_name["eta_vs_c_floor.png"].batch_id == "Start202609301533"
    assert by_name["eta_vs_c_floor.png"].condition_id is None
    assert by_name["env.jpg"].batch_id == "Start202609301533"
    assert by_name["env.jpg"].condition_id == "202609301533"
    assert by_name["all_batches_comparison.png"].batch_id == "trainlog"       # root の名前
    assert Path(by_name["env.jpg"].path).is_absolute()
    assert result.attachments_for({"Start202609301533"}) == [by_name["eta_vs_c_floor.png"], by_name["env.jpg"]]
    assert result.attachments_for({"nothing"}) == []


def test_zip_only_batch_is_reported_as_skipped(tree: Path):
    result = load_all(tree)
    assert any(s.path.endswith("Start202609281636.zip") for s in result.skipped)


def test_old_layout_condition_dir_directly_under_root(tmp_path: Path):
    root = tmp_path / "trainlog"
    cond = root / "202606051724"
    cond.mkdir(parents=True)
    (cond / "config.yaml").write_text(CONFIG)
    (cond / "train_log_01.txt").write_text(_log(1))
    run = load_all(root).runs[0]
    assert run.batch_id == "202606051724" and run.condition_id == "202606051724"
    assert run.started_at == datetime(2026, 6, 5, 17, 24)
    assert run.hints == []


def test_config_yaml_is_fallback_when_args_line_unreadable(tmp_path: Path):
    cond = tmp_path / "Start202601010000" / "202601010000"
    cond.mkdir(parents=True)
    (cond / "config_01.yaml").write_text(CONFIG)
    (cond / "train_log_01.txt").write_text(_log(2, args="not a dict line"))
    run = load_all(tmp_path).runs[0]
    assert run.algorithm == "SRS_v2_v3"
    assert any("config.yaml" in w for w in run.warnings)


def test_broken_config_is_reported_but_runs_still_load(tmp_path: Path):
    cond = tmp_path / "Start202601010000" / "202601010000"
    cond.mkdir(parents=True)
    (cond / "config_01.yaml").write_text("train_args: [unclosed\n")
    (cond / "train_log_01.txt").write_text(_log(1))
    result = load_all(tmp_path)
    assert len(result.runs) == 1
    assert any("config_01.yaml" in s.reason for s in result.skipped)


def test_missing_path_and_empty_dir_return_errors(tmp_path: Path):
    assert load_all(tmp_path / "nope").errors[0].reason == "パスが存在しません"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "train_log" in load_all(empty).errors[0].reason


def test_sample_data_loads_if_present():
    sample = Path(__file__).resolve().parent.parent / "sample_data" / "trainlog"
    if not sample.exists():
        pytest.skip("sample_data が未生成 (python scripts/make_sample_data.py)")
    result = load_all(sample)
    assert len(result.runs) == 24
    assert len(result.errors) == 2
    assert {r.algorithm for r in result.runs} == {"BASE", "RSRS", "SRS_v2_v3", "SRS_v2_c_floor"}
    assert {a.name for a in result.attachments} == {
        "SRS_v2_v3実装η比較.png", "eta_vs_c_floor.png", "simple_pyramid.png", "SRS_v2_v3_eta_0.1-1.0_comparison.png",
    }
