from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_regime_dashboard_chart(df: pd.DataFrame):
    work = df.copy().sort_values("date")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.60, 0.40],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(
            x=work["date"],
            y=work["indiavix"],
            mode="lines",
            name="INDIAVIX",
            line=dict(color="#2b6cb0", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(43, 108, 176, 0.12)",
            hovertemplate="INDIAVIX: %{y:.2f}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    if "prob_regime_high_vol" in work.columns:
        storm = work["indiavix"].where(work["prob_regime_high_vol"] >= 0.5)
        fig.add_trace(
            go.Scatter(
                x=work["date"],
                y=storm,
                mode="lines",
                name="High-vol window",
                line=dict(color="#c53030", width=2.4),
                hovertemplate="Storm VIX: %{y:.2f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=work["date"],
                y=work["prob_regime_high_vol"],
                mode="lines",
                name="P(high vol)",
                line=dict(color="#e53e3e", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(197, 48, 48, 0.14)",
                hovertemplate="P(high): %{y:.1%}<extra></extra>",
                showlegend=True,
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

    if "macro_sentiment" in work.columns:
        fig.add_trace(
            go.Scatter(
                x=work["date"],
                y=work["macro_sentiment"],
                mode="lines",
                name="RBI stance",
                line=dict(color="#276749", width=1.5),
                hovertemplate="Sentiment: %{y:+.2f}<extra></extra>",
                showlegend=True,
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

    last = work.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[last["date"]],
            y=[last["indiavix"]],
            mode="markers+text",
            name="Last print",
            marker=dict(color="#1a365d", size=9, line=dict(color="white", width=1.2)),
            text=[f"{float(last['indiavix']):.1f}"],
            textposition="middle right",
            textfont=dict(size=11, color="#1a365d"),
            hovertemplate="Last %{x|%Y-%m-%d}<br>VIX %{y:.2f}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title=dict(text="INDIAVIX with high-vol windows and RBI stance", x=0.01, xanchor="left"),
        template="plotly_white",
        paper_bgcolor="#f4f6f8",
        plot_bgcolor="#ffffff",
        height=720,
        margin=dict(l=72, r=64, t=88, b=56),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(26,54,93,0.15)",
            borderwidth=1,
            font=dict(size=12, color="#1a202c"),
            itemsizing="constant",
        ),
        hovermode="x unified",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1a202c"),
        showlegend=True,
    )

    fig.update_yaxes(
        title_text="India VIX",
        showticklabels=True,
        ticks="outside",
        tickfont=dict(size=11, color="#1a202c"),
        title_font=dict(size=13, color="#1a202c"),
        gridcolor="rgba(26,54,93,0.12)",
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="P(high vol)",
        range=[0, 1],
        tickformat=".0%",
        showticklabels=True,
        ticks="outside",
        tickfont=dict(size=11, color="#c53030"),
        title_font=dict(size=13, color="#c53030"),
        gridcolor="rgba(197,48,48,0.10)",
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="RBI sentiment",
        showticklabels=True,
        ticks="outside",
        tickfont=dict(size=11, color="#276749"),
        title_font=dict(size=13, color="#276749"),
        zeroline=True,
        zerolinecolor="rgba(39,103,73,0.25)",
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.update_xaxes(
        title_text="",
        showticklabels=True,
        ticks="outside",
        tickformat="%b %Y",
        tickfont=dict(size=11, color="#1a202c"),
        gridcolor="rgba(26,54,93,0.08)",
        showgrid=True,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Date",
        showticklabels=True,
        ticks="outside",
        tickformat="%b %Y",
        tickfont=dict(size=11, color="#1a202c"),
        title_font=dict(size=13, color="#1a202c"),
        gridcolor="rgba(26,54,93,0.08)",
        showgrid=True,
        rangeslider=dict(visible=True, thickness=0.06),
        row=2,
        col=1,
    )
    return fig