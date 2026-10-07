"""RL Experiment Tracker - Streamlit アプリ本体。

起動:  streamlit run app.py   (uv を使うなら  uv run streamlit run app.py)

画面構成:
  サイドバー : データの場所、評価指標、条件キー、絞り込み
  一覧       : 読み込んだ全ランの表
  比較       : 選んだ条件/ランの学習曲線・最終値・パラメータ差分
  集計       : 条件ごとの実行回数・seed・平均・標準偏差・不足判定
  図         : Start... フォルダに保存されている png をバッチごとに閲覧
  読み込み状況: 失敗したファイルと理由、未対応の物、警告付きのラン

このファイルは「状態 (選択値) を集めて rl_tracker の関数を呼び、結果を表示する」だけにし、
計算は rl_tracker/ 側に置いている。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rl_tracker import __version__
from rl_tracker.aggregate import (
    DEFAULT_IGNORED_KEYS,
    MISSING,
    batch_overview,
    condition_label,
    group_by_condition,
    missing_parameters,
    parameter_diff,
    short_key,
    summarize_conditions,
    varying_parameters,
)
from rl_tracker.charts import MAX_SERIES, condition_bars, learning_curves, mean_band
from rl_tracker.filters import RunFilter, apply_filter, parameter_values
from rl_tracker.frame import curves_to_frame, mean_curves_to_frame, runs_to_frame, summaries_to_frame
from rl_tracker.loaders import load_all
from rl_tracker.models import (
    DEFAULT_SCORE_METRIC,
    STATUS_FINISHED,
    STATUS_INCOMPLETE,
    Attachment,
    ExperimentRun,
    LoadResult,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = APP_DIR / "sample_data" / "trainlog"

# 指標の表示名。ここに無い指標 (loss.xxx など) はキーをそのまま表示する
METRIC_LABELS = {
    "average_reward": "average reward (評価対局の平均報酬)",
    "win_rate": "win rate (評価対局の勝率)",
    "generation_mean": "generation stats mean (学習中リターン平均)",
    "generation_std": "generation stats std",
}

st.set_page_config(page_title="RL Experiment Tracker", layout="wide")


# ---------------------------------------------------------------------------
# データ読み込み (キャッシュ)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="バックアップを読み込んでいます...")
def load_cached(root: str, _reload_token: int) -> LoadResult:
    """load_all の結果をパスごとにキャッシュする。_reload_token を変えると再読み込みされる。"""
    return load_all(root)


def metric_options(runs: list[ExperimentRun]) -> list[str]:
    """読み込んだランに存在する指標名。主要な物を先頭に、loss.* は後ろに。"""
    seen: set[str] = set()
    for r in runs:
        seen.update(r.metrics)
    head = [m for m in METRIC_LABELS if m in seen]
    tail = sorted(m for m in seen if m not in METRIC_LABELS)
    return head + tail


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


# ---------------------------------------------------------------------------
# サイドバー
# ---------------------------------------------------------------------------
def sidebar() -> tuple[LoadResult, list[ExperimentRun], str, list[str], int]:
    """サイドバーを描画し、(読み込み結果, 絞り込み後のラン, 指標, 条件キー, 必要ラン数) を返す。"""
    st.sidebar.title("RL Experiment Tracker")
    st.sidebar.caption(f"v{__version__} · ローカル専用 · 元ファイルは読み取りのみ")

    root = st.sidebar.text_input(
        "バックアップのディレクトリ",
        value=str(DEFAULT_ROOT),
        help="HandyRL の trainlog ディレクトリ (Start... フォルダが並んでいる場所) を指定します",
    )
    st.session_state["root_path"] = root  # 図タブで trainlog 直下の図を見分けるために使う
    if "reload_token" not in st.session_state:
        st.session_state.reload_token = 0
    if st.sidebar.button("再読み込み", help="ファイルが増えた時に押してください"):
        st.session_state.reload_token += 1

    result = load_cached(root, st.session_state.reload_token)
    runs = result.runs
    n_err = len(result.errors)
    st.sidebar.markdown(
        f"**{len(runs)} ラン** を読み込み · "
        + (f":red[失敗 {n_err} 件]" if n_err else "失敗 0 件")
        + (f" · 未対応 {len(result.skipped)} 件" if result.skipped else "")
    )
    if not runs:
        return result, [], DEFAULT_SCORE_METRIC, [], 10

    # --- 指標と条件キー ----------------------------------------------------
    st.sidebar.divider()
    metrics = metric_options(runs)
    default_idx = metrics.index(DEFAULT_SCORE_METRIC) if DEFAULT_SCORE_METRIC in metrics else 0
    metric = st.sidebar.selectbox("最終評価値に使う指標", metrics, index=default_idx, format_func=metric_label)

    all_keys = sorted({k for r in runs for k in r.parameters} - DEFAULT_IGNORED_KEYS)
    auto_keys = varying_parameters(runs)
    condition_keys = st.sidebar.multiselect(
        "条件を区別するハイパーパラメータ",
        options=all_keys,
        default=auto_keys,
        format_func=short_key,
        help="既定では、読み込んだ実験の間で値が異なるパラメータを自動で選びます。"
             "アルゴリズム (agent.type) は常に条件に含まれます",
    )
    min_runs = st.sidebar.number_input("1 条件あたり必要な完了ラン数", min_value=1, max_value=100, value=10,
                                       help="集計画面で、この数に満たない条件を『不足』として示します")

    # --- 絞り込み ----------------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("絞り込み")
    f = RunFilter()
    f.algorithms = set(st.sidebar.multiselect("アルゴリズム", sorted({str(r.algorithm) for r in runs})))
    f.seeds = set(st.sidebar.multiselect("seed", sorted({str(r.seed) for r in runs})))
    f.statuses = set(st.sidebar.multiselect("実行状態", [STATUS_FINISHED, STATUS_INCOMPLETE]))
    f.batches = set(st.sidebar.multiselect("バッチ (Start...)", sorted({r.batch_id for r in runs}, reverse=True)))

    param_filter_keys = st.sidebar.multiselect(
        "ハイパーパラメータで絞り込む", options=all_keys, default=[], format_func=short_key,
    )
    for key in param_filter_keys:
        chosen = st.sidebar.multiselect(
            f"{short_key(key)} の値", options=parameter_values(runs, key), key=f"pf_{key}",
        )
        if chosen:
            f.parameters[key] = set(chosen)

    filtered = apply_filter(runs, f)
    st.sidebar.caption(f"絞り込み後: {len(filtered)} ラン")
    return result, filtered, metric, condition_keys, int(min_runs)


# ---------------------------------------------------------------------------
# 一覧
# ---------------------------------------------------------------------------
def page_list(runs: list[ExperimentRun], metric: str, condition_keys: list[str], result: LoadResult) -> None:
    st.subheader(f"実験一覧 ({len(runs)} ラン)")
    if not runs:
        st.info("表示できるランがありません。絞り込み条件を見直してください。")
        return

    df = runs_to_frame(runs, condition_keys, metric)
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "final_score": st.column_config.NumberColumn(f"最終値 ({short_metric(metric)})", format="%.3f"),
            "started_at": st.column_config.DatetimeColumn("実行日時", format="YYYY-MM-DD HH:mm"),
            "duration_min": st.column_config.NumberColumn("所要 (分)", format="%.1f"),
            "source_path": st.column_config.TextColumn("元ファイル"),
            "warnings": st.column_config.NumberColumn("警告数"),
        },
        height=min(600, 40 + 35 * len(df)),
    )
    st.download_button("この表を CSV で保存", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="experiments.csv", mime="text/csv")

    with st.expander("1 ランの詳細を見る (全パラメータ・警告・ヒント)"):
        chosen = st.selectbox("run_id", [r.run_id for r in runs])
        run = next(r for r in runs if r.run_id == chosen)
        c1, c2 = st.columns([2, 1])
        with c1:
            params = pd.DataFrame(
                [(k, str(v)) for k, v in sorted(run.parameters.items()) if k not in DEFAULT_IGNORED_KEYS or k == "train_args.seed"],
                columns=["パラメータ", "値"],
            )
            st.dataframe(params, hide_index=True, height=400)
        with c2:
            st.markdown(f"**状態**: {run.status} · **エポック**: {run.epochs} · **エピソード**: {run.episodes}")
            st.markdown(f"**元ファイル**: `{run.source_path}`")
            st.markdown(f"**config**: `{run.config_path or '—'}`")
            st.markdown(f"**git commit**: {run.git_commit or '— (バックアップに記録なし)'}")
            st.markdown(f"**目的 / メモ**: {run.purpose or '— (記録なし)'}")
            for w in run.warnings:
                st.warning(w)
        batch_images = result.attachments_for({run.batch_id})
        if batch_images:
            st.markdown(f"**同じバッチ ({run.batch_id}) にある図** — 実験の目的を推測する手がかり")
            render_images(batch_images, columns=2)


def short_metric(metric: str) -> str:
    return metric_label(metric).split(" (")[0]


# ---------------------------------------------------------------------------
# 比較
# ---------------------------------------------------------------------------
def page_compare(runs: list[ExperimentRun], metric: str, condition_keys: list[str], result: LoadResult) -> None:
    st.subheader("実験比較")
    if len(runs) < 2:
        st.info("比較には 2 ラン以上必要です。")
        return

    groups = group_by_condition(runs, condition_keys)
    labels = {condition_label(g[0], condition_keys): g for g in groups.values()}
    label_list = sorted(labels)

    mode = st.radio("比較の単位", ["条件ごと (平均 ± 標準偏差)", "ラン個別"], horizontal=True)
    opts = metric_options(runs)
    curve_metric = st.selectbox("学習曲線に表示する指標", opts,
                                index=opts.index(metric) if metric in opts else 0, format_func=metric_label)

    if mode.startswith("条件"):
        chosen = st.multiselect("比較する条件", label_list, default=label_list[: min(2, len(label_list))],
                                max_selections=MAX_SERIES)
        if len(chosen) < 2:
            st.info("条件を 2 つ以上選んでください。")
            return
        selected_groups = {lab: labels[lab] for lab in chosen}
        selected_runs = [r for lab in chosen for r in labels[lab]]
        representatives = [labels[lab][0] for lab in chosen]
        col_names = {rep.run_id: lab for rep, lab in zip(representatives, chosen)}

        df = mean_curves_to_frame(selected_groups, curve_metric)
        if df.empty:
            st.warning(f"指標 {curve_metric} の値がありません。")
        else:
            st.altair_chart(mean_band(df, short_metric(curve_metric), chosen))
            st.caption("帯は ±1 標本標準偏差。途中で止まったランは、値のあるエポックだけ平均に含まれます (n はツールチップ参照)。")
    else:
        run_ids = [r.run_id for r in runs]
        chosen_ids = st.multiselect("比較するラン", run_ids, default=run_ids[:2])
        if len(chosen_ids) < 2:
            st.info("ランを 2 つ以上選んでください。")
            return
        selected_runs = [r for r in runs if r.run_id in chosen_ids]
        representatives = selected_runs
        col_names = {r.run_id: r.run_id for r in selected_runs}
        chosen = sorted({condition_label(r, condition_keys) for r in selected_runs})
        if len(chosen) > MAX_SERIES:
            st.warning(f"条件が {len(chosen)} 種類あります。色で区別できるのは {MAX_SERIES} 種類までなので、条件を絞ってください。")
            return
        df = curves_to_frame(selected_runs, curve_metric, condition_keys)
        if df.empty:
            st.warning(f"指標 {curve_metric} の値がありません。")
        else:
            st.altair_chart(learning_curves(df, short_metric(curve_metric), chosen))

    # --- 最終値・平均・標準偏差 -------------------------------------------
    st.markdown("#### 最終評価値と実行回数")
    summaries = summarize_conditions(selected_runs, condition_keys, metric)
    sdf = summaries_to_frame(summaries)[["label", "n_runs", "n_finished", "n_seeds", "seeds", "mean", "std", "min", "max"]]
    st.dataframe(sdf.rename(columns={"label": "条件", "n_runs": "実行回数", "n_finished": "完了", "n_seeds": "seed数",
                                     "seeds": "使用seed", "mean": "平均", "std": "標準偏差", "min": "最小", "max": "最大"}),
                 hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["平均", "標準偏差", "最小", "最大"]})
    st.caption(f"指標: {metric_label(metric)} の最終エポック値。平均・標準偏差は完了ランのみ (標準偏差は ddof=1、1 件では未定義)。")

    with st.expander("ラン個別の最終値"):
        rows = [{"run_id": r.run_id, "条件": condition_label(r, condition_keys), "最終値": r.final_value(metric),
                 "状態": r.status, "エポック": r.epochs} for r in selected_runs]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"最終値": st.column_config.NumberColumn(format="%.3f")})

    # --- パラメータ差分と欠損 -----------------------------------------------
    st.markdown("#### ハイパーパラメータの差分")
    diff = parameter_diff(representatives)
    if not diff:
        st.success("選択した対象の間でハイパーパラメータに差はありません (seed・サーバー設定を除く)。")
    else:
        table = pd.DataFrame(
            [{"パラメータ": short_key(k), **{col_names[rid]: str(v) for rid, v in vals.items()}} for k, vals in diff.items()]
        )
        st.dataframe(table, hide_index=True)

    missing = missing_parameters(representatives)
    if missing:
        st.markdown("#### 欠損している条件")
        st.warning("一部の対象にしか存在しないパラメータがあります。実装や設定項目が異なる実験同士の比較になっていないか確認してください。")
        for k, absent in missing.items():
            st.markdown(f"- `{short_key(k)}` が無い: {', '.join(col_names.get(a, a) for a in absent)}")

    # --- 関係するバッチの図 -----------------------------------------------
    related = result.attachments_for({r.batch_id for r in selected_runs})
    if related:
        with st.expander(f"比較対象のバッチに保存されている図 ({len(related)} 枚)"):
            render_images(related, columns=2)


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
def page_summary(runs: list[ExperimentRun], metric: str, condition_keys: list[str], min_runs: int) -> None:
    st.subheader("条件ごとの集計")
    if not runs:
        st.info("表示できるランがありません。")
        return

    include_incomplete = st.checkbox("未完了ランも平均に含める", value=False,
                                     help="既定では途中で止まったランは実行回数には数えますが、平均・標準偏差からは除きます")
    summaries = summarize_conditions(runs, condition_keys, metric, include_incomplete=include_incomplete)
    df = summaries_to_frame(summaries)
    df.insert(0, "不足", df["n_finished"] < min_runs)

    only_short = st.checkbox(f"完了ランが {min_runs} 回に満たない条件だけ表示", value=False)
    shown = df[df["不足"]] if only_short else df
    st.dataframe(
        shown.drop(columns=["label"]),
        hide_index=True,
        column_config={
            "不足": st.column_config.CheckboxColumn("不足", help=f"完了ランが {min_runs} 回未満"),
            "n_runs": "実行回数", "n_finished": "完了", "n_seeds": "seed数", "seeds": "使用seed",
            "mean": st.column_config.NumberColumn("平均", format="%.3f"),
            "std": st.column_config.NumberColumn("標準偏差", format="%.3f"),
            "min": st.column_config.NumberColumn("最小", format="%.3f"),
            "max": st.column_config.NumberColumn("最大", format="%.3f"),
            "n_values": "平均に使った数", "batches": "バッチ",
        },
    )
    st.caption(f"指標: {metric_label(metric)}。seed は train_args.seed の値 (SRS_v2 の実験では全て 0 のため、run 番号で繰り返しを区別しています)。")

    if not shown.empty and shown["mean"].notna().any():
        st.altair_chart(condition_bars(shown, short_metric(metric)))

    st.markdown("#### 条件を選んで個別ランを見る")
    label_list = [s.label for s in summaries]
    chosen = st.selectbox("条件", label_list)
    group = next(s for s in summaries if s.label == chosen)
    members = [r for r in runs if r.run_id in set(group.run_ids)]
    st.markdown(f"**{group.n_runs} 回実行** (完了 {group.n_finished}) · seed: {', '.join(map(str, group.seeds))} · バッチ: {', '.join(group.batches)}")
    st.dataframe(
        runs_to_frame(members, [], metric)[["run_id", "seed", "started_at", "final_score", "status", "epochs", "duration_min", "source_path"]],
        hide_index=True,
        column_config={
            "final_score": st.column_config.NumberColumn(f"最終値 ({short_metric(metric)})", format="%.3f"),
            "started_at": st.column_config.DatetimeColumn("実行日時", format="YYYY-MM-DD HH:mm"),
            "duration_min": st.column_config.NumberColumn("所要 (分)", format="%.1f"),
        },
    )


# ---------------------------------------------------------------------------
# 図 (バッチに付随する png)
# ---------------------------------------------------------------------------
def render_images(attachments: list[Attachment], columns: int = 2) -> None:
    """画像をグリッドで表示する。読めないファイルはその場でエラー表示し、他の画像は表示を続ける。"""
    if not attachments:
        st.caption("図はありません。")
        return
    cols = st.columns(columns)
    for i, a in enumerate(attachments):
        with cols[i % columns]:
            caption = a.name if a.condition_id is None else f"{a.name}  ({a.condition_id} 内)"
            try:
                st.image(a.path, caption=caption)
            except Exception as exc:  # noqa: BLE001 - 壊れた画像 1 枚で画面を止めない
                st.error(f"{a.name} を表示できません: {type(exc).__name__}: {exc}")


def page_figures(result: LoadResult, runs: list[ExperimentRun], condition_keys: list[str]) -> None:
    st.subheader("図 (バッチに保存されている画像)")
    st.caption(
        "`Start...` フォルダや trainlog 直下に手作業で置いた png/jpg をそのまま表示します。"
        "ファイルは読み取るだけで、コピーや移動はしません。画像にカーソルを合わせると右上に拡大ボタンが出ます。"
    )
    if not result.attachments:
        st.info("画像ファイルが見つかりませんでした。")
        return

    # 絞り込み後のランが属するバッチ + trainlog 直下 (バッチに属さない比較図) を既定で表示する
    run_batches = {r.batch_id for r in runs}
    all_batches = sorted({a.batch_id for a in result.attachments}, reverse=True)
    root_name = Path(st.session_state.get("root_path", "")).name
    default_batches = [b for b in all_batches if b in run_batches or b == root_name]

    chosen = st.multiselect("表示するバッチ", all_batches, default=default_batches,
                            help="既定ではサイドバーの絞り込みに一致するランを含むバッチだけを表示します")
    columns = st.radio("列数", [1, 2, 3], index=1, horizontal=True)

    overview = batch_overview(runs, condition_keys)
    shown = 0
    for batch_id in chosen:
        images = [a for a in result.attachments if a.batch_id == batch_id]
        if not images:
            continue
        shown += len(images)
        st.markdown(f"#### {batch_id}")
        conditions = overview.get(batch_id)
        if conditions:
            st.markdown("含まれる条件: " + " / ".join(f"`{label}` ×{n}" for label, n in conditions))
        elif batch_id == root_name:
            st.caption("trainlog 直下の図 (複数バッチをまたぐ比較図など)")
        else:
            st.caption("このバッチのランは現在の絞り込みに含まれていません")
        render_images(images, columns=int(columns))
        st.divider()
    if shown == 0:
        st.info("選択したバッチに画像はありません。")


# ---------------------------------------------------------------------------
# 読み込み状況
# ---------------------------------------------------------------------------
def page_status(result: LoadResult) -> None:
    st.subheader("読み込み状況")
    c1, c2, c3 = st.columns(3)
    c1.metric("読み込めたラン", len(result.runs))
    c2.metric("失敗したファイル", len(result.errors))
    c3.metric("未対応・注意", len(result.skipped))

    st.markdown("#### 読み込みに失敗したファイル")
    if result.errors:
        st.dataframe(pd.DataFrame([{"ファイル": e.path, "理由": e.reason} for e in result.errors]), hide_index=True)
    else:
        st.success("失敗したファイルはありません。")

    st.markdown("#### 読み込まなかった物・注意")
    if result.skipped:
        st.dataframe(pd.DataFrame([{"対象": s.path, "理由": s.reason} for s in result.skipped]), hide_index=True)
    else:
        st.write("ありません。")

    warned = [r for r in result.runs if r.warnings]
    st.markdown(f"#### 警告付きで読み込んだラン ({len(warned)})")
    if warned:
        st.dataframe(pd.DataFrame([{"run_id": r.run_id, "警告": " / ".join(r.warnings)} for r in warned]), hide_index=True)
    else:
        st.write("ありません。")

    st.markdown("#### このバックアップ形式で取得できない項目")
    st.markdown(
        "- **seed**: `train_args.seed` のみ記録 (全実験で 0)。torch の seed は固定されていないため run 間の違いは run 番号で区別\n"
        "- **Git コミット ID**: ログ・config に記録なし\n"
        "- **実験目的・メモ**: 構造化データなし。バッチ内の画像 (図タブ) が唯一の手がかり\n"
        "- **環境ステップ数**: HandyRL はエポック / エピソード単位で記録"
    )


# ---------------------------------------------------------------------------
def main() -> None:
    result, runs, metric, condition_keys, min_runs = sidebar()

    if not result.runs:
        st.title("RL Experiment Tracker")
        if result.errors:
            for e in result.errors:
                st.error(f"{e.path}: {e.reason}")
        st.info("サイドバーでバックアップのディレクトリを指定してください。サンプルは `sample_data/trainlog` にあります。")
        return

    tab_list, tab_compare, tab_summary, tab_figures, tab_status = st.tabs(["一覧", "比較", "集計", "図", "読み込み状況"])
    with tab_list:
        page_list(runs, metric, condition_keys, result)
    with tab_compare:
        page_compare(runs, metric, condition_keys, result)
    with tab_summary:
        page_summary(runs, metric, condition_keys, min_runs)
    with tab_figures:
        page_figures(result, runs, condition_keys)
    with tab_status:
        page_status(result)


main()
