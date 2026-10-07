# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

SRS_v2 (HandyRL) の実験バックアップ `trainlog/` を読み、バッチごとに「保存した図」と「各条件を何回回したか (= 何回の平均か)」を表示する最小の Streamlit アプリ。コード・コメント・UI 文字列はすべて日本語。

## コマンド

```bash
uv sync --extra dev                         # 初回 (pip なら pip install -r requirements.txt)
uv run streamlit run app.py                 # 起動 (.claude/launch.json は port 8501, headless)
uv run pytest -q                            # テスト (test_scan.py のみ)
uv run pytest -q test_scan.py::test_run_count_equals_number_of_train_logs   # 単体
```

`pyproject.toml` で `testpaths = ["test_scan.py"]` にしてあり、`archive_v1/tests/` は意図的に収集しない。lint/formatter の設定は無い。

## 構成

トップレベルの 3 ファイルだけが現行コード。

- `scan.py` — Streamlit 非依存の走査ロジック。`scan(root)` が `(list[Batch], root直下の画像)` を返す。`Batch` → `Condition` の 2 階層。
- `app.py` — `scan()` を `st.cache_data` 越しに呼んで表示するだけ。ロジックは `scan.py` 側に置く。
- `test_scan.py` — `tmp_path` に小さな trainlog を作って `scan.py` を検証。

`archive_v1/` は以前の多機能版 (`rl_tracker/` パッケージ: loaders/parsers/aggregate/charts、学習曲線・集計) の凍結コピー。現行コードからは参照しない。学習曲線や条件横断集計が必要になったときの参考用。

## 走査ロジックの要点 (scan.py)

- **条件フォルダ** = `train_log_*.txt` を含むフォルダ (`rglob` で検出)。その親が**バッチ** (`Start...`)。親が root 自身の場合 (旧形式: `trainlog/202606051724/`) は条件フォルダ自身をバッチとして扱う。
- **実行回数** = `train_log_*.txt` のファイル本数。途中終了ログも含む。
- **正常終了** = ログ末尾 300 バイトに `time :` があるか (HandyRL が終了時に出力)。巨大ログ対策で末尾のみ読む。
- **設定** = `train_log` の 1 行目 (Python dict 表現) を `ast.literal_eval` で読み、`a.b.c` 形式に平坦化。読めなければ `{}` → UI では「不明」。推測で埋めない。
- `differing_keys()` がバッチ内で値の異なる設定キー (η など) を返し、表の列になる。`IGNORED_KEYS` (サーバーアドレス等) は除外。`short()` が表示用にキーの前置きを削る。
- バッチは名前 (= `YYYYMMDDHHMM` を含む) の降順で並べる。zip だけ残っているバッチは対象外。

## 制約

- バックアップは**読み取り専用**。書き込み・移動・外部送信をするコードを足さない。
- 既定のデータパスは `~/Desktop/SRS_v2_backup/trainlog`。無ければ `sample_data/trainlog` (合成データ) にフォールバック。
- seed は全実験 0 (研究コードの仕様) なので表示しない。
