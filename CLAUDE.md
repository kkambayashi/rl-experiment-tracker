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

ユーザーは普段 `アプリを起動.command` を Finder でダブルクリックして起動する (port 8501, `--server.address localhost`。起動済みならブラウザで開くだけ)。動作確認で別に起動するときは 8502 など別ポートを使い、終わったら止める。lint/formatter の設定は無い。

## 構成

- `scan.py` — Streamlit 非依存の走査ロジック。`scan(root)` が `(list[Batch], root直下の画像)` を返す。`Batch` → `Condition` の 2 階層。
- `app.py` — 画面。上から「バッチ一覧 (1 行 = 1 バッチの `st.dataframe`、`on_select="rerun"` で 1 行選択)」→「選んだバッチの図と条件ごとの表 (`show_batch`、未選択なら最新)」→「trainlog 直下の図」。一覧の行は `batches` と同じ順で作るので、選択行の番号がそのまま `batches` の添字になる。
- `test_scan.py` — `tmp_path` に小さな trainlog を作って `scan.py` を検証。
- `アプリを起動.command` — 上記のダブルクリック起動用スクリプト。

以前の多機能版 (`archive_v1/rl_tracker/`: loaders/parsers/aggregate/charts、学習曲線・集計) は削除済み。最初のコミット `dd3ead9` に残っているので、必要なら `git show dd3ead9:archive_v1/<path>` で参照する。

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
