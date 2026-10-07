"""RL Experiment Tracker (最小版)

起動:  streamlit run app.py      (uv なら  uv run streamlit run app.py)

できること:
  1. Start... フォルダに保存した png をまとめて見る
  2. 各条件フォルダの train_log_XX.txt の本数 = 何回の平均か を確認する
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from scan import Batch, differing_keys, scan, short

APP_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = Path.home() / "Desktop" / "SRS_v2_backup" / "trainlog"   # 実データ
if not DEFAULT_ROOT.exists():
    DEFAULT_ROOT = APP_DIR / "sample_data" / "trainlog"                 # 無ければサンプル

st.set_page_config(page_title="RL Experiment Tracker", layout="wide")


@st.cache_data(show_spinner="trainlog を読み込み中...")
def load(root: str) -> tuple[list[Batch], list[Path]]:
    return scan(root)


# ---------------------------------------------------------------- サイドバー
st.sidebar.title("RL Experiment Tracker")
root = st.sidebar.text_input("trainlog フォルダのパス", value=str(DEFAULT_ROOT))
if st.sidebar.button("再読み込み", help="実験を追加した後に押す"):
    st.cache_data.clear()
query = st.sidebar.text_input("フォルダ名で絞り込み", placeholder="例: 202609")
columns = st.sidebar.radio("画像の列数", [1, 2, 3], index=1, horizontal=True)
show_settings = st.sidebar.checkbox("条件の違い (η など) を表に出す", value=True)

batches, root_images = load(root)
if not batches:
    st.error(f"train_log_*.txt が見つかりません: {root}")
    st.stop()

if query:
    batches = [b for b in batches if query in b.name or any(query in c.name for c in b.conditions)]
st.sidebar.caption(f"{len(batches)} バッチ · {sum(b.total_runs for b in batches)} ラン")

if not batches:
    st.info("該当するバッチがありません")
    st.stop()



# ---------------------------------------------------------------- 本文
def show_images(paths: list[Path], n_cols: int) -> None:
    cols = st.columns(n_cols)
    for i, p in enumerate(paths):
        with cols[i % n_cols]:
            try:
                st.image(str(p), caption=p.name)
            except Exception as exc:  # noqa: BLE001 - 壊れた画像 1 枚で止めない
                st.error(f"{p.name} を表示できません: {exc}")


def show_runs(batch: Batch) -> None:
    """条件フォルダごとの実行回数の表。設定の違いがあれば列として足す。"""
    diff = differing_keys(batch.conditions) if show_settings else []
    rows = []
    for c in batch.conditions:
        row = {"条件フォルダ": c.name, "アルゴリズム": c.algorithm}
        for key in diff:
            if key != "train_args.agent.type":  # アルゴリズム列と重複するので除く
                row[short(key)] = str(c.settings.get(key, "—"))
        row["実行回数 (= 平均の回数)"] = c.n_runs
        row["正常終了"] = c.n_finished
        rows.append(row)
    st.dataframe(rows, hide_index=True)
    incomplete = sum(c.n_runs - c.n_finished for c in batch.conditions)
    if incomplete:
        st.warning(f"{incomplete} 本のログは途中で終わっています (末尾に time: が無い)。平均に含めるか注意してください。")

summary = []
for b in batches:
    summary.append({
        "日時": b.started_at.strftime("%Y-%m-%d %H:%M") if b.started_at else "",
        "バッチ": b.name,
        "アルゴリズム": " / ".join(sorted({c.algorithm for c in b.conditions})),
        "比べている設定": ", ".join(short(k) for k in differing_keys(b.conditions) if k != "train_args.agent.type"),
        "条件数": len(b.conditions),
        "平均回数": " / ".join(str(n) for n in sorted({c.n_runs for c in b.conditions})),
    })

st.header("バッチ一覧")
event = st.dataframe(summary, hide_index=True, on_select="rerun", selection_mode="single-row", column_config={"条件数": st.column_config.NumberColumn(alignment="left")},)



def show_batch(batch: Batch) -> None:
    when = batch.started_at.strftime("%Y-%m-%d %H:%M") if batch.started_at else ""
    st.header(f"{batch.name}  ·  {when}")
    runs = " / ".join(f"{c.n_runs}回" for c in batch.conditions)
    st.caption(f"{len(batch.conditions)} 条件 · 実行回数 {runs} · {batch.path}")
    if batch.images:
        show_images(batch.images, columns)
    else:
        st.caption("図はありません")
    show_runs(batch)
    st.divider()

selected = event.selection.rows
if selected:
    batch = batches[selected[0]]   # 選ばれた行と同じ番号のバッチ
else:
    batch = batches[0]             # 何も選んでいなければ最新
show_batch(batch)

if root_images:
    st.header("trainlog 直下の図")
    show_images(root_images, columns)
    st.divider()
