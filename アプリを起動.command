#!/bin/zsh
# ダブルクリックで RL Experiment Tracker を起動し、ブラウザで開く。
# 止めるときは、このターミナルのウィンドウを閉じる (または Ctrl+C)。

export PATH="$HOME/.local/bin:$PATH"   # uv の場所 (Finder から起動しても見つかるように)
cd "$(dirname "$0")"                   # このファイルがあるフォルダ (= プロジェクト) へ移動
URL="http://localhost:8501"

# すでに起動していれば、ブラウザで開くだけ (2 回ダブルクリックしても大丈夫なように)
if curl -s "$URL/_stcore/health" >/dev/null 2>&1; then
  open "$URL"
  exit 0
fi

# --server.address localhost: この Mac からしか開けないようにする
uv run streamlit run app.py --server.address localhost --server.port 8501
