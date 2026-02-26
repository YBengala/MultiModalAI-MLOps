import os
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from components.styles import apply_custom_css
from sqlalchemy import create_engine

apply_custom_css()

DB_HOST = os.getenv("RAKUTEN_DB_HOST", "localhost")
DB_PORT = os.getenv("RAKUTEN_DB_PORT", "5432")
DB_NAME = os.getenv("RAKUTEN_DB_NAME", "rakuten_db")
DB_USER = os.getenv("RAKUTEN_DB_USER", "")
DB_PASSWORD = os.getenv("RAKUTEN_DB_PASSWORD", "")

st.header("📊 Monitoring")
st.markdown("Real-time tracking of production predictions.")


def get_db_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


@st.cache_data(ttl=30)
def load_predictions():
    try:
        engine = get_db_engine()
        df = pd.read_sql(
            """
            SELECT id, designation, predicted_class, confidence, has_image, dt_predicted
            FROM predictions_prod
            ORDER BY dt_predicted DESC
            LIMIT 200
            """,
            engine,
        )
        return df, None
    except Exception as e:
        if "UndefinedTable" in str(e) or "does not exist" in str(e):
            return None, "table_missing"
        return None, str(e)


col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

df, error = load_predictions()

if error == "table_missing":
    st.info(
        "Table `predictions_prod` does not exist yet. "
        "Restart the PostgreSQL container to apply SQL migrations, "
        "then make predictions from the **Prediction Demo** page."
    )
elif error:
    st.error(f"Database connection error: {error}")
elif df is None or df.empty:
    st.info(
        "No production predictions yet. "
        "Go to the **Prediction Demo** page to classify products."
    )
else:
    paris = ZoneInfo("Europe/Paris")

    df["dt_predicted"] = (
        pd.to_datetime(df["dt_predicted"]).dt.tz_localize("UTC").dt.tz_convert(paris)
    )

    # Global metrics
    total = len(df)
    dominant_class = df["predicted_class"].value_counts().idxmax()
    avg_confidence = df["confidence"].mean()
    high_conf = (df["confidence"] >= 0.75).sum()
    pct_high = high_conf / total

    def _kpi(label: str, value: str) -> str:
        return (
            f'<div style="margin-bottom:8px">'
            f'<div style="font-size:0.85rem;color:#94A3B8;margin-bottom:6px">{label}</div>'
            f'<div style="font-size:2rem;font-weight:700;color:#E2E8F0;line-height:1.3">{value}</div>'
            f"</div>"
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi("Total Predictions", str(total)), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi("Dominant Class", dominant_class), unsafe_allow_html=True)
    with c3:
        st.markdown(
            _kpi("Avg Confidence", f"{avg_confidence:.1%}"), unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            _kpi("High Confidence (≥75%)", f"{pct_high:.0%}"), unsafe_allow_html=True
        )

    st.divider()

    # charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Predicted Class Distribution")
        top10 = df["predicted_class"].value_counts().head(10).reset_index()
        top10.columns = ["Class", "Count"]
        fig_dist = px.bar(
            top10,
            x="Count",
            y="Class",
            orientation="h",
            color="Count",
            color_continuous_scale=[[0, "#1E3A5F"], [0.5, "#38BDF8"], [1, "#7DD3FC"]],
            text="Count",
        )
        fig_dist.update_traces(textposition="outside", marker_line_width=0)
        fig_dist.update_layout(
            plot_bgcolor="#1A2332",
            paper_bgcolor="#1A2332",
            font_color="#E2E8F0",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            margin={"t": 20, "b": 20, "l": 10, "r": 40},
            height=350,
        )
        st.plotly_chart(fig_dist, width='stretch')

    with col_right:
        st.subheader("Confidence Distribution")
        fig_hist = px.histogram(
            df,
            x="confidence",
            nbins=20,
            color_discrete_sequence=["#38BDF8"],
            labels={"confidence": "Confidence", "count": "Count"},
        )
        fig_hist.update_layout(
            plot_bgcolor="#1A2332",
            paper_bgcolor="#1A2332",
            font_color="#E2E8F0",
            bargap=0.05,
            margin={"t": 20, "b": 20},
            height=350,
            xaxis={"tickformat": ".0%"},
        )
        fig_hist.add_vrect(
            x0=0.75,
            x1=1.0,
            fillcolor="#16A34A",
            opacity=0.08,
            line_width=0,
            annotation_text="High",
            annotation_position="top right",
            annotation_font_color="#16A34A",
        )
        fig_hist.add_vrect(
            x0=0.5,
            x1=0.75,
            fillcolor="#D97706",
            opacity=0.08,
            line_width=0,
            annotation_text="Mid",
            annotation_position="top right",
            annotation_font_color="#D97706",
        )
        st.plotly_chart(fig_hist, width='stretch')

    st.divider()

    # Predictions over time
    st.subheader("Predictions Over Time")
    df_time = df.copy()
    df_time["hour"] = df_time["dt_predicted"].dt.floor("h")
    time_agg = (
        df_time.groupby("hour")
        .agg(
            count=("id", "count"),
            avg_conf=("confidence", "mean"),
        )
        .reset_index()
    )

    fig_time = go.Figure()
    fig_time.add_trace(
        go.Bar(
            x=time_agg["hour"],
            y=time_agg["count"],
            name="Predictions",
            marker_color="#38BDF8",
            opacity=0.8,
        )
    )
    fig_time.add_trace(
        go.Scatter(
            x=time_agg["hour"],
            y=time_agg["avg_conf"],
            name="Avg Confidence",
            yaxis="y2",
            line={"color": "#F59E0B", "width": 2},
            mode="lines+markers",
        )
    )
    fig_time.update_layout(
        plot_bgcolor="#1A2332",
        paper_bgcolor="#1A2332",
        font_color="#E2E8F0",
        legend={"orientation": "h", "y": 1.1},
        yaxis={"title": "Predictions", "gridcolor": "#243447"},
        yaxis2={
            "title": "Avg Confidence",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
            "range": [0, 1],
        },
        margin={"t": 20, "b": 20},
        height=280,
    )
    st.plotly_chart(fig_time, width='stretch')

    st.divider()

    # Predictions table
    st.subheader("Latest Predictions")

    df_display = df.copy()
    df_display["dt_predicted"] = df_display["dt_predicted"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    df_display["designation"] = df_display["designation"].str[:80] + "…"

    df_display[""] = df_display["confidence"].apply(
        lambda v: "🟢" if v >= 0.75 else ("🟡" if v >= 0.5 else "🔴")
    )
    df_display["confidence"] = df_display["confidence"].apply(lambda v: f"{v:.1%}")

    st.dataframe(
        df_display[
            [
                "",
                "dt_predicted",
                "predicted_class",
                "confidence",
                "has_image",
                "designation",
            ]
        ].rename(
            columns={
                "dt_predicted": "Date (Paris)",
                "predicted_class": "Predicted Class",
                "confidence": "Confidence",
                "has_image": "Image",
                "designation": "Designation",
            }
        ),
        width='stretch',
        hide_index=True,
    )
