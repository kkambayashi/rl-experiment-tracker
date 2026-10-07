# 自分で手を入れるための小課題

コードの流れを理解するために、小さい順に並べています。各課題はテストを 1 つ足してから実装すると安全です。

## 1. 一覧に「所要時間 (時間)」の列を足す  — 難易度: 低

- 触るファイル: `rl_tracker/frame.py` の `runs_to_frame`
- 今は `duration_min` (分) だけ。`duration_hour` を追加し、`app.py` の `column_config` に表示名を加える。
- 確認: `tests/test_loaders.py` の `test_run_fields_from_directory_layout` の近くに、`runs_to_frame` の列を確認するテストを追加。

## 2. 「評価値が初めて 0.9 を超えたエポック」(t90) を指標に加える — 難易度: 中

- 触るファイル: `rl_tracker/aggregate.py` (関数を 1 つ追加)、`app.py` (集計表に列を追加)
- `analysis_2026xxxx/*_per_run_metrics.csv` で手作業で計算していた `t90` と同じ物。
- `ExperimentRun.metrics["average_reward"]` を先頭から見て、初めて 0.9 以上になった index を返す。見つからなければ `None` (推測しない)。
- 確認: `tests/test_aggregate.py` に、閾値を超えない系列で `None` になるテストを書く。

## 3. zip のみのバッチを読むローダーを追加する — 難易度: 中〜高

- 触るファイル: `rl_tracker/loaders/handyrl_zip.py` (新規)、`rl_tracker/loaders/__init__.py` の `LOADERS`
- `zipfile.ZipFile(path).namelist()` で `train_log_*.txt` を探し、`read()` した文字列を既存の `parse_train_log` に渡す。
  ディレクトリ版 (`handyrl_dir.py`) の `_build_run` と同じ組み立てになるので、共通部分を関数に切り出すと重複が減る。
- 展開済みディレクトリがある zip は重複なので読まない (今の `_collect_zip_only_batches` のロジックを参考に)。
- 確認: `tests/test_loaders.py` に、`zipfile` で tmp_path 内に zip を作って読むテストを追加。

## 4. 研究コード側で Git コミット ID を残す — 難易度: 低 (ただし研究コードの変更なので要相談)

- `bash_scripts/experiment_config.sh` の `mkdir trainlog/Start$StartDATE/$DATE` の直後に
  `git rev-parse HEAD > trainlog/Start$StartDATE/$DATE/git_commit.txt` を 1 行足すだけ。
- アプリ側は `handyrl_dir.py` の `_load_condition_dir` で `git_commit.txt` があれば読んで `ExperimentRun.git_commit` に入れる。
  無ければ今まで通り `None`。
