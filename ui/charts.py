# ui/charts.py
import plotly.graph_objects as go


# ── Shared dark theme ─────────────────────────────────────────────────────────

DARK = dict(
    paper_bgcolor="#1a1d27",
    plot_bgcolor="#1a1d27",
    font=dict(color="#7b8099", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#2a2e45", zerolinecolor="#2a2e45"),
    yaxis=dict(gridcolor="#2a2e45", zerolinecolor="#2a2e45"),
)

COLORS = {
    "blue":   "#4f8ef7",
    "green":  "#34d17a",
    "red":    "#f05b5b",
    "amber":  "#f5a623",
    "muted":  "#7b8099",
    "border": "#2a2e45",
}


def empty_chart(message: str = "No data") -> go.Figure:
    """Blank placeholder chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=COLORS["muted"], size=13),
    )
    fig.update_layout(title=message, **DARK)
    return fig


def trend_chart(trends: dict) -> go.Figure:
    """Bar chart showing avg % change per column."""
    if not trends:
        return empty_chart("No trend data available")

    cols   = list(trends.keys())
    values = [trends[c]["avg_pct_change"] for c in cols]
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cols,
        y=values,
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["muted"]),
    ))
    fig.update_layout(
        title="Average % Change by Column",
        showlegend=False,
        **DARK,
    )
    return fig


def correlation_chart(correlations: dict) -> go.Figure:
    """Horizontal bar chart of feature correlations."""
    if not correlations:
        return empty_chart("No strong correlations found")

    pairs  = list(correlations.keys())[:8]
    values = [correlations[p] for p in pairs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=pairs,
        orientation="h",
        marker_color=[COLORS["blue"] if v >= 0 else COLORS["red"] for v in values],
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["muted"]),
    ))
    fig.update_layout(
        title="Strong Feature Correlations",
        xaxis=dict(
            range=[-1, 1],
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["blue"],
        ),
        yaxis=dict(gridcolor=COLORS["border"]),
        paper_bgcolor=DARK["paper_bgcolor"],
        plot_bgcolor=DARK["plot_bgcolor"],
        font=DARK["font"],
        margin=DARK["margin"],
        showlegend=False,
    )
    return fig


def anomaly_chart(anomalies: list) -> go.Figure:
    """Bar chart of anomaly counts per column."""
    fig = go.Figure()

    if not anomalies:
        fig.add_annotation(
            text="No anomalies detected",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=COLORS["green"], size=14),
        )
        fig.update_layout(title="Anomaly Detection", **DARK)
        return fig

    cols   = [a["column"] for a in anomalies]
    counts = [a["count"]  for a in anomalies]

    fig.add_trace(go.Bar(
        x=cols,
        y=counts,
        marker_color=COLORS["amber"],
        text=counts,
        textposition="outside",
        textfont=dict(size=11, color=COLORS["muted"]),
    ))
    fig.update_layout(
        title="Anomalies Detected per Column",
        showlegend=False,
        **DARK,
    )
    return fig


def coefficient_chart(coefficients: dict) -> go.Figure:
    """Bar chart of regression feature coefficients."""
    if not coefficients:
        return empty_chart("No coefficients available")

    sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    feats  = [c[0] for c in sorted_coefs]
    values = [c[1] for c in sorted_coefs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feats,
        y=values,
        marker_color=[COLORS["blue"] if v >= 0 else COLORS["red"] for v in values],
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["muted"]),
    ))
    fig.update_layout(
        title="Feature Coefficients (standardized)",
        showlegend=False,
        **DARK,
    )
    return fig


def r2_gauge(r2: float) -> go.Figure:
    """Gauge chart showing R² score with color coding."""
    color = (
        COLORS["green"] if r2 > 0.8
        else COLORS["amber"] if r2 > 0.5
        else COLORS["red"]
    )
    quality = (
        "Strong fit" if r2 > 0.8
        else "Moderate fit" if r2 > 0.5
        else "Weak fit"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=r2,
        title=dict(
            text=f"R² Score — {quality}",
            font=dict(color=COLORS["muted"], size=13),
        ),
        number=dict(font=dict(color=color, size=40)),
        gauge=dict(
            axis=dict(
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                tickcolor=COLORS["muted"],
                tickfont=dict(color=COLORS["muted"]),
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#21253a",
            borderwidth=1,
            bordercolor=COLORS["border"],
            steps=[
                dict(range=[0.0, 0.5], color="#2a1a1a"),
                dict(range=[0.5, 0.8], color="#2a2a1a"),
                dict(range=[0.8, 1.0], color="#1a2a1a"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.75,
                value=r2,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#1a1d27",
        font=dict(color=COLORS["muted"]),
        margin=dict(l=30, r=30, t=60, b=20),
        height=260,
    )
    return fig


def price_history_chart(dates: list, prices: list, ticker: str = "") -> go.Figure:
    """Line chart for stock price history."""
    if not dates or not prices:
        return empty_chart("No price data available")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode="lines",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.08)",
        name=ticker,
    ))
    fig.update_layout(
        title=f"{ticker} Price History" if ticker else "Price History",
        showlegend=False,
        **DARK,
    )
    return fig


def multi_line_chart(data: dict, title: str = "") -> go.Figure:
    """Multiple line series on one chart. data = {label: [values]}"""
    if not data:
        return empty_chart("No data available")

    palette = [COLORS["blue"], COLORS["green"], COLORS["amber"], COLORS["red"]]
    fig = go.Figure()

    for i, (label, values) in enumerate(data.items()):
        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="lines+markers",
            name=label,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(
            bgcolor="#1a1d27",
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["muted"]),
        ),
        **DARK,
    )
    return fig