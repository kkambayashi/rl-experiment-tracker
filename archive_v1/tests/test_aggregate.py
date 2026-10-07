"""aggregate.py のテスト。期待値は手計算できる小さな数で書く。"""

import math

import pytest

from rl_tracker.aggregate import (
    MISSING,
    batch_overview,
    condition_key,
    condition_label,
    group_by_condition,
    mean_curve,
    mean_std,
    missing_parameters,
    parameter_diff,
    short_key,
    summarize_conditions,
    varying_parameters,
)
from rl_tracker.models import STATUS_FINISHED, STATUS_INCOMPLETE, ExperimentRun


def make_run(run_id: str, algorithm: str, params: dict, rewards: list, status: str = STATUS_FINISHED,
             seed: int | None = 0, batch: str = "B1") -> ExperimentRun:
    """テスト用の最小 ExperimentRun。metrics は average_reward だけ持つ。"""
    full = {"train_args.agent.type": algorithm, "train_args.seed": seed,
            "worker_args.server_address": "10.0.0.1", **params}
    return ExperimentRun(
        run_id=run_id, batch_id=batch, condition_id=run_id.split("/")[0], run_index=1,
        algorithm=algorithm, variant=algorithm, seed=seed, started_at=None, status=status,
        epochs=len(rewards), episodes=None, duration_sec=None, env="simple_pyramid",
        parameters=full, metrics={"average_reward": rewards}, source_path=f"/x/{run_id}.txt",
    )


ETA = "train_args.metadata.global_eta"
CF = "train_args.metadata.c_floor"

RUNS = [
    make_run("c1/01", "SRS_v2_v3", {ETA: 0.1}, [0.0, 0.5, 1.0]),
    make_run("c1/02", "SRS_v2_v3", {ETA: 0.1}, [0.0, 0.4, 0.8]),
    make_run("c1/03", "SRS_v2_v3", {ETA: 0.1}, [0.0, 0.3], status=STATUS_INCOMPLETE),   # 途中で停止
    make_run("c2/01", "SRS_v2_v3", {ETA: 0.5}, [0.0, 0.2, 0.6], batch="B2"),
    make_run("c3/01", "SRS_v2_c_floor", {ETA: 0.1, CF: 0.001}, [0.0, 0.6, 0.9]),
    make_run("c4/01", "BASE", {}, [0.0, 0.1, 0.2], seed=1),
]


# --- mean_std -----------------------------------------------------------------
def test_mean_std_basic():
    mean, std, n = mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0 and n == 3
    assert std == pytest.approx(1.0)  # 標本標準偏差 (ddof=1): sqrt(((1+0+1))/2) = 1


def test_mean_std_ignores_none_and_nan():
    mean, std, n = mean_std([1.0, None, float("nan"), 3.0])
    assert mean == 2.0 and n == 2
    assert std == pytest.approx(math.sqrt(2))


def test_mean_std_edge_cases():
    assert mean_std([]) == (None, None, 0)
    assert mean_std([None]) == (None, None, 0)
    assert mean_std([5.0]) == (5.0, None, 1)  # 1 件では std を定義しない


# --- 条件キー -------------------------------------------------------------------
def test_varying_parameters_excludes_ignored_and_presence_only_by_default():
    keys = varying_parameters(RUNS)
    assert keys == [ETA]                      # c_floor は値が 1 種類なので入らない
    assert "train_args.seed" not in keys      # seed は条件ではない
    assert "worker_args.server_address" not in keys


def test_varying_parameters_include_presence():
    assert varying_parameters(RUNS, include_presence=True) == [CF, ETA]


def test_condition_key_and_label():
    run = RUNS[4]
    assert condition_key(run, [ETA, CF]) == ("SRS_v2_c_floor", 0.1, 0.001)
    assert condition_key(RUNS[0], [ETA, CF]) == ("SRS_v2_v3", 0.1, MISSING)
    assert condition_label(RUNS[0], [ETA, CF]) == "SRS_v2_v3 | global_eta=0.1 | c_floor=—"


def test_short_key():
    assert short_key("train_args.metadata.global_eta") == "global_eta"
    assert short_key("train_args.policy_target") == "policy_target"
    assert short_key("env_args.param.depth") == "depth"
    assert short_key("something.else") == "something.else"


def test_group_by_condition_counts_runs_across_batches():
    groups = group_by_condition(RUNS, [ETA])
    assert len(groups[("SRS_v2_v3", 0.1)]) == 3
    assert len(groups[("SRS_v2_v3", 0.5)]) == 1
    assert len(groups[("BASE", MISSING)]) == 1


# --- summarize_conditions -------------------------------------------------------
def test_summarize_conditions_excludes_incomplete_by_default():
    s = {x.key: x for x in summarize_conditions(RUNS, [ETA])}
    c1 = s[("SRS_v2_v3", 0.1)]
    assert c1.n_runs == 3 and c1.n_finished == 2 and c1.n_values == 2
    assert c1.mean == pytest.approx(0.9)          # (1.0 + 0.8) / 2
    assert c1.std == pytest.approx(math.sqrt(((0.1) ** 2 + (0.1) ** 2) / 1))  # 0.1414
    assert c1.min == 0.8 and c1.max == 1.0
    assert c1.seeds == [0] and c1.n_seeds == 1
    assert c1.batches == ["B1"]


def test_summarize_conditions_include_incomplete():
    s = {x.key: x for x in summarize_conditions(RUNS, [ETA], include_incomplete=True)}
    c1 = s[("SRS_v2_v3", 0.1)]
    assert c1.n_values == 3
    assert c1.mean == pytest.approx((1.0 + 0.8 + 0.3) / 3)


def test_summarize_single_run_has_no_std():
    s = {x.key: x for x in summarize_conditions(RUNS, [ETA])}
    assert s[("SRS_v2_v3", 0.5)].std is None
    assert s[("SRS_v2_v3", 0.5)].mean == pytest.approx(0.6)


def test_summarize_seed_list_distinct():
    runs = [make_run("a/1", "X", {}, [1.0], seed=3), make_run("a/2", "X", {}, [1.0], seed=3),
            make_run("a/3", "X", {}, [1.0], seed=7)]
    (s,) = summarize_conditions(runs, [])
    assert s.seeds == [3, 7] and s.n_seeds == 2 and s.n_runs == 3


# --- mean_curve -----------------------------------------------------------------
def test_mean_curve_handles_different_lengths():
    epochs, means, stds, counts = mean_curve(RUNS[:3], "average_reward")
    assert epochs == [0, 1, 2]
    assert means == [pytest.approx(0.0), pytest.approx(0.4), pytest.approx(0.9)]
    assert counts == [3, 3, 2]  # 3 本目は 2 エポックで止まっている
    assert stds[1] == pytest.approx(0.1)


def test_mean_curve_empty():
    assert mean_curve([], "average_reward") == ([], [], [], [])


# --- 比較 -----------------------------------------------------------------------
def test_parameter_diff_only_differing_keys():
    diff = parameter_diff([RUNS[0], RUNS[3], RUNS[4]])
    assert set(diff) == {ETA, CF}
    assert diff[ETA] == {"c1/01": 0.1, "c2/01": 0.5, "c3/01": 0.1}
    assert diff[CF]["c1/01"] == MISSING and diff[CF]["c3/01"] == 0.001


def test_missing_parameters_lists_runs_without_key():
    missing = missing_parameters([RUNS[0], RUNS[4]])
    assert missing == {CF: ["c1/01"]}
    assert missing_parameters([RUNS[0], RUNS[1]]) == {}


# --- バッチ ---------------------------------------------------------------------
def test_batch_overview_counts_conditions_per_batch():
    overview = batch_overview(RUNS, [ETA])
    assert set(overview) == {"B1", "B2"}
    assert overview["B2"] == [("SRS_v2_v3 | global_eta=0.5", 1)]
    assert ("SRS_v2_v3 | global_eta=0.1", 3) in overview["B1"]
    assert ("BASE | global_eta=—", 1) in overview["B1"]
