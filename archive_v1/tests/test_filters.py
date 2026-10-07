from rl_tracker.filters import RunFilter, apply_filter, parameter_values
from rl_tracker.models import STATUS_FINISHED, STATUS_INCOMPLETE
from tests.test_aggregate import ETA, CF, make_run

RUNS = [
    make_run("a/1", "SRS_v2_v3", {ETA: 0.1}, [1.0]),
    make_run("a/2", "SRS_v2_v3", {ETA: 0.5}, [0.5], status=STATUS_INCOMPLETE),
    make_run("b/1", "BASE", {}, [0.2], seed=1, batch="B2"),
    make_run("c/1", "SRS_v2_c_floor", {ETA: 0.1, CF: 0.001}, [0.9]),
]


def test_empty_filter_keeps_everything():
    assert apply_filter(RUNS, RunFilter()) == RUNS


def test_filter_by_algorithm_and_status():
    out = apply_filter(RUNS, RunFilter(algorithms={"SRS_v2_v3"}, statuses={STATUS_FINISHED}))
    assert [r.run_id for r in out] == ["a/1"]


def test_filter_by_seed_uses_strings():
    assert [r.run_id for r in apply_filter(RUNS, RunFilter(seeds={"1"}))] == ["b/1"]


def test_filter_by_parameter_including_missing():
    out = apply_filter(RUNS, RunFilter(parameters={ETA: {"0.1"}}))
    assert [r.run_id for r in out] == ["a/1", "c/1"]
    out = apply_filter(RUNS, RunFilter(parameters={CF: {"—"}}))
    assert [r.run_id for r in out] == ["a/1", "a/2", "b/1"]


def test_filter_by_batch():
    assert [r.run_id for r in apply_filter(RUNS, RunFilter(batches={"B2"}))] == ["b/1"]


def test_parameter_values_sorted_numeric_then_missing():
    assert parameter_values(RUNS, ETA) == ["0.1", "0.5", "—"]
    assert parameter_values(RUNS, CF) == ["0.001", "—"]
