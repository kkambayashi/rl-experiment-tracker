"""Altair のグラフ定義。Streamlit に依存しないので単体で確認できる。

色は「系列の並び順に固定」で割り当てる (フィルタで系列が減っても色が変わらないように、
呼び出し側で domain の順序を決めて渡す)。8 色を超える比較は想定しない (読めなくなる)。
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# 色覚多様性に配慮して検証済みの 8 色 (順序も含めて固定)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MAX_SERIES = len(PALETTE)


def _color_scale(domain: list[str]) -> alt.Scale:
    return alt.Scale(domain=domain, range=PALETTE[: len(domain)])


# Streamlit は既定で autosize=fit (軸や凡例込みで height に収める) にするため、
# 凡例が大きいと描画領域が潰れる。fit-x (幅だけ合わせる) を明示して height を描画領域の高さにする。
_AUTOSIZE = alt.AutoSizeParams(type="fit-x", contains="padding")


def learning_curves(df: pd.DataFrame, metric_label: str, domain: list[str]) -> alt.LayerChart:
    """ラン 1 本ごとの学習曲線。色は条件、線はラン。

    df の列: run_id, condition, epoch, value
    """
    base = alt.Chart(df).encode(
        x=alt.X("epoch:Q", title="epoch"),
        y=alt.Y("value:Q", title=metric_label),
        color=alt.Color("condition:N", scale=_color_scale(domain), title="条件",
                        legend=alt.Legend(orient="bottom", columns=1, labelLimit=600)),
        detail="run_id:N",
    )
    lines = base.mark_line(strokeWidth=1.5, opacity=0.85)

    # マウスに最も近い epoch を選んで縦線とツールチップを出す
    hover = alt.selection_point(fields=["epoch"], nearest=True, on="mouseover", empty=False)
    points = base.mark_circle(size=60).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("run_id:N", title="run"),
            alt.Tooltip("condition:N", title="条件"),
            alt.Tooltip("epoch:Q"),
            alt.Tooltip("value:Q", title=metric_label, format=".3f"),
        ],
    ).add_params(hover)
    rule = alt.Chart(df).mark_rule(color="#999").encode(x="epoch:Q").transform_filter(hover)

    return alt.layer(lines, points, rule).properties(width="container", height=380, autosize=_AUTOSIZE)


def mean_band(df: pd.DataFrame, metric_label: str, domain: list[str]) -> alt.LayerChart:
    """条件ごとの平均曲線 ± 標準偏差の帯。

    df の列: condition, epoch, mean, std, lower, upper, n
    """
    color = alt.Color("condition:N", scale=_color_scale(domain), title="条件",
                      legend=alt.Legend(orient="bottom", columns=1, labelLimit=600))
    band = alt.Chart(df).mark_area(opacity=0.18).encode(
        x=alt.X("epoch:Q", title="epoch"),
        y=alt.Y("lower:Q", title=metric_label),
        y2="upper:Q",
        color=color,
    )
    line = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x="epoch:Q", y=alt.Y("mean:Q", title=metric_label), color=color,
    )
    hover = alt.selection_point(fields=["epoch"], nearest=True, on="mouseover", empty=False)
    points = alt.Chart(df).mark_circle(size=70).encode(
        x="epoch:Q", y="mean:Q", color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("condition:N", title="条件"),
            alt.Tooltip("epoch:Q"),
            alt.Tooltip("mean:Q", title=f"{metric_label} 平均", format=".3f"),
            alt.Tooltip("std:Q", title="標準偏差", format=".3f"),
            alt.Tooltip("n:Q", title="ラン数"),
        ],
    ).add_params(hover)
    rule = alt.Chart(df).mark_rule(color="#999").encode(x="epoch:Q").transform_filter(hover)
    return alt.layer(band, line, points, rule).properties(width="container", height=380, autosize=_AUTOSIZE)


def condition_bars(df: pd.DataFrame, metric_label: str) -> alt.LayerChart:
    """集計画面: 条件ごとの最終値の平均を横棒で、標準偏差をエラーバーで示す。

    df の列: label, mean, std, n_values (std が None の行はエラーバー無し)
    """
    data = df.dropna(subset=["mean"]).copy()
    data["lo"] = data["mean"] - data["std"].fillna(0)
    data["hi"] = data["mean"] + data["std"].fillna(0)
    y = alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420))
    bars = alt.Chart(data).mark_bar(color=PALETTE[0], cornerRadiusEnd=4, size=18).encode(
        x=alt.X("mean:Q", title=f"{metric_label} の平均 (最終値)"),
        y=y,
        tooltip=[
            alt.Tooltip("label:N", title="条件"),
            alt.Tooltip("mean:Q", format=".3f", title="平均"),
            alt.Tooltip("std:Q", format=".3f", title="標準偏差"),
            alt.Tooltip("n_values:Q", title="平均に使ったラン数"),
        ],
    )
    error = alt.Chart(data).mark_rule(color="#333", strokeWidth=1.5).encode(x="lo:Q", x2="hi:Q", y=y)
    height = max(120, 32 * len(data))
    return alt.layer(bars, error).properties(width="container", height=height, autosize=_AUTOSIZE)
