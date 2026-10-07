"""parsers.py のテスト。文字列を渡して構造を確認するだけなのでファイル不要。"""

import pytest

from rl_tracker.parsers import (
    epochs_to_metrics,
    flatten,
    parse_args_line,
    parse_config_yaml,
    parse_train_log,
    scalarize,
)

ARGS_LINE = (
    "{'env_args': {'env': 'simple_pyramid', 'param': {'depth': 6}}, "
    "'train_args': {'default_learning_rate': 3e-08, 'seed': 0, "
    "'agent': {'type': 'SRS_v2_v3', 'use_RND': False}, "
    "'metadata': {'name': ['knn', 'global_eta'], 'knn': {'size': 10000, 'k': 64}, 'global_eta': 0.1}}, "
    "'worker_args': {'server_address': '0.0.0.0', 'num_parallel': 128}}"
)

FINISHED_LOG = f"""{ARGS_LINE}
waiting training
started server
50 100 150 200 start sender
start receiver
started training
250 300 350
epoch 0
win rate = 0.500 (2189.0 / 4378)
average reward = 0.000 (2189.0 / 4378)
generation stats = 0.071 +- 0.257
loss = p:0.126 v:0.042 q:0.039 entropy_srs:0.397 ent:1.382 total:0.206
updated model(89)
400 450 500
epoch 1
win rate = 0.656 (84.0 / 128)
average reward = 0.312 (84.0 / 128)
generation stats = 0.007 +- 0.081
loss = p:0.011 v:0.018 q:0.034 entropy_srs:0.389 ent:1.366 total:0.061
updated model(172)
550 600
epoch 2
win rate = Nan (0)
generation stats = Nan (0)
loss = p:-0.001 v:0.016 total:0.044
updated model(257)
closed worker 0
disconnected
finished server
time : 3096.573837242089
"""


def test_parse_args_line_restores_nested_dict():
    args = parse_args_line(ARGS_LINE)
    assert args["train_args"]["agent"]["type"] == "SRS_v2_v3"
    assert args["train_args"]["default_learning_rate"] == 3e-08
    assert args["train_args"]["metadata"]["knn"]["k"] == 64


def test_parse_args_line_rejects_non_dict_and_garbage():
    assert parse_args_line("waiting training") is None
    assert parse_args_line("{'a': __import__('os')}") is None  # literal_eval はコードを評価しない
    assert parse_args_line("[1, 2, 3]") is None


def test_parse_train_log_finished():
    parsed = parse_train_log(FINISHED_LOG)
    assert parsed.args is not None
    assert parsed.finished is True
    assert parsed.duration_sec == pytest.approx(3096.5738, rel=1e-6)
    assert parsed.max_episode_count == 600
    assert [e.epoch for e in parsed.epochs] == [0, 1, 2]
    assert parsed.warnings == []


def test_parse_train_log_epoch_values():
    e0, e1, e2 = parse_train_log(FINISHED_LOG).epochs
    assert e0.win_rate == 0.5 and e0.eval_games == 4378
    assert e1.average_reward == pytest.approx(0.312)
    assert e1.generation_mean == pytest.approx(0.007)
    assert e1.generation_std == pytest.approx(0.081)
    assert e1.losses == {"p": 0.011, "v": 0.018, "q": 0.034, "entropy_srs": 0.389, "ent": 1.366, "total": 0.061}
    assert e1.model_steps == 172
    # Nan の行は None のまま (0 にしない)
    assert e2.win_rate is None and e2.average_reward is None and e2.generation_mean is None
    assert "q" not in e2.losses  # loss のキーはエポックごとに増減しうる


def test_parse_train_log_incomplete_has_no_time():
    text = FINISHED_LOG.split("closed worker 0")[0]
    parsed = parse_train_log(text)
    assert parsed.finished is False
    assert parsed.duration_sec is None
    assert len(parsed.epochs) == 3


def test_parse_train_log_args_only():
    parsed = parse_train_log(ARGS_LINE + "\n")
    assert parsed.args is not None
    assert parsed.epochs == []
    assert parsed.finished is False


def test_parse_train_log_garbage_does_not_raise():
    parsed = parse_train_log("\x00\xff not a log\nstill not\n")
    assert parsed.args is None
    assert parsed.epochs == []
    assert any("1行目" in w for w in parsed.warnings)


def test_parse_train_log_win_rate_with_opponent_tags():
    text = f"""{ARGS_LINE}
epoch 0
win rate (total) = 0.700 (70.0 / 100)
average reward (total) = 0.400 (70.0 / 100)
win rate (random) = 0.800 (40.0 / 50)
average reward (random) = 0.600 (40.0 / 50)
"""
    e0 = parse_train_log(text).epochs[0]
    assert e0.win_rate == pytest.approx(0.7)
    assert e0.average_reward == pytest.approx(0.4)
    assert e0.win_rate_by_opponent == {"random": pytest.approx(0.8)}


def test_epochs_to_metrics_aligns_variable_loss_keys():
    metrics = epochs_to_metrics(parse_train_log(FINISHED_LOG).epochs)
    assert metrics["win_rate"] == [0.5, pytest.approx(0.656), None]
    assert metrics["loss.q"] == [0.039, 0.034, None]
    assert metrics["loss.total"] == [0.206, 0.061, 0.044]
    assert set(metrics) >= {"win_rate", "average_reward", "generation_mean", "generation_std"}


def test_parse_config_yaml_and_flatten():
    text = """
env_args:
    env: 'simple_pyramid'   # コメント
    param:
        depth: 6
train_args:
    default_learning_rate: 3.0e-8
    agent:
        type: 'SRS_v2'
    metadata:
        name: ['knn', 'global_eta']
        global_eta: 0.1
"""
    flat = flatten(parse_config_yaml(text))
    assert flat["env_args.env"] == "simple_pyramid"
    assert flat["train_args.default_learning_rate"] == pytest.approx(3e-8)
    assert flat["train_args.agent.type"] == "SRS_v2"
    assert flat["train_args.metadata.name"] == ["knn", "global_eta"]
    assert flat["train_args.metadata.global_eta"] == 0.1


def test_parse_config_yaml_rejects_non_dict():
    with pytest.raises(ValueError):
        parse_config_yaml("- just\n- a list\n")


def test_scalarize_turns_lists_into_strings():
    assert scalarize([[1, 5]]) == "[[1, 5]]"
    assert scalarize(0.1) == 0.1
    assert scalarize("x") == "x"
