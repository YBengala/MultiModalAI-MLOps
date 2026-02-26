import os
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st
from components.minio_client import fetch_image
from components.styles import apply_custom_css
from sqlalchemy import create_engine

apply_custom_css()

DB_HOST = os.getenv("RAKUTEN_DB_HOST", "localhost")
DB_PORT = os.getenv("RAKUTEN_DB_PORT", "5432")
DB_NAME = os.getenv("RAKUTEN_DB_NAME", "rakuten_db")
DB_USER = os.getenv("RAKUTEN_DB_USER", "")
DB_PASSWORD = os.getenv("RAKUTEN_DB_PASSWORD", "")


def get_db_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


st.header("🗂️ Product Gallery by Class")


@st.cache_data(ttl=300)
def load_class_list():
    try:
        engine = get_db_engine()
        df = pd.read_sql("SELECT DISTINCT prdtypecode, prodtype FROM products_processed ORDER BY prodtype", engine)
        return df
    except Exception:
        return pd.DataFrame(columns=["prdtypecode", "prodtype"])


@st.cache_data(ttl=300)
def load_global_stats():
    try:
        engine = get_db_engine()
        stats = pd.read_sql(
            """
            SELECT
                COUNT(*)                                        AS total_products,
                COUNT(DISTINCT prdtypecode)                     AS total_classes,
                ROUND(100.0 * SUM(CASE WHEN image_exists THEN 1 ELSE 0 END) / COUNT(*), 1)
                                                                AS pct_with_image,
                MAX(batch_id)                                   AS last_batch,
                MAX(dt_processed)                               AS last_batch_date
            FROM products_latest
            """,
            engine,
        )
        return stats.iloc[0].to_dict()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_class_distribution():
    try:
        engine = get_db_engine()
        df = pd.read_sql(
            "SELECT prodtype, COUNT(*) as nb FROM products_latest GROUP BY prodtype ORDER BY nb DESC",
            engine,
        )
        return df
    except Exception:
        return pd.DataFrame(columns=["prodtype", "nb"])


@st.cache_data(ttl=60)
def load_products_for_class(prdtypecode: int):
    try:
        engine = get_db_engine()
        df = pd.read_sql(
            "SELECT * FROM products_latest WHERE prdtypecode = %(code)s LIMIT 12",
            engine,
            params={"code": prdtypecode},
        )
        return df
    except Exception:
        return pd.DataFrame()


classes_df = load_class_list()

if classes_df.empty:
    st.info("No data available. Please run the ingestion pipeline first.")
else:
    stats = load_global_stats()
    dist_df_early = load_class_distribution()
    imbalance_ratio = (
        round(dist_df_early["nb"].max() / dist_df_early["nb"].min(), 1)
        if not dist_df_early.empty and dist_df_early["nb"].min() > 0
        else None
    )

    if stats:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Products", f"{int(stats.get('total_products', 0)):,}".replace(",", " "))
        c2.metric("Classes", int(stats.get("total_classes", 0)))
        c3.metric("Image Coverage", f"{stats.get('pct_with_image', 0):.0f} %")
        c4.metric("Imbalance Ratio", f"{imbalance_ratio}×" if imbalance_ratio else "—")
        c5.metric("Latest Batch", stats.get("last_batch", "—"))

        last_date = stats.get("last_batch_date")
        if last_date:
            date_str = last_date.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M")
            st.caption(f"Last ingestion : {date_str} (Paris)")

    st.divider()

    dist_df = dist_df_early

    if not dist_df.empty:
        fig = px.bar(
            dist_df,
            x="nb",
            y="prodtype",
            orientation="h",
            color="nb",
            color_continuous_scale=[[0, "#1E3A5F"], [0.5, "#38BDF8"], [1, "#7DD3FC"]],
            labels={"nb": "Number of Products", "prodtype": "Class"},
            text="nb",
        )
        fig.update_traces(textposition="outside", textfont_size=11, marker_line_width=0)
        fig.update_layout(
            title={"text": "Product Distribution by Class", "font": {"size": 15}, "x": 0},
            plot_bgcolor="#1A2332",
            paper_bgcolor="#1A2332",
            font_color="#E2E8F0",
            coloraxis_showscale=False,
            height=620,
            margin={"t": 40, "b": 10, "l": 10, "r": 60},
            yaxis={"categoryorder": "total ascending", "tickfont": {"size": 11}},
            xaxis={"gridcolor": "#243447"},
        )
        st.plotly_chart(fig, width='stretch')

    st.divider()

    class_options = {row["prodtype"]: row["prdtypecode"] for _, row in classes_df.iterrows()}
    selected_class = st.selectbox("Select a class", options=list(class_options.keys()))
    selected_code = class_options[selected_class]

    products_df = load_products_for_class(selected_code)

    if products_df.empty:
        st.info(f"No products found for class « {selected_class} ».")
    else:
        st.caption(f"{len(products_df)} products displayed")
        cols = st.columns(3)
        for idx, (_, row) in enumerate(products_df.iterrows()):
            with cols[idx % 3]:
                img = None
                path = row.get("path_image_minio")
                exists = row.get("image_exists")
                if path and exists:
                    img = fetch_image(str(path))

                if img:
                    st.image(img, width=250)
                else:
                    st.markdown(
                        '<div style="height:120px;background:#243447;border-radius:6px;display:flex;'
                        'align-items:center;justify-content:center;color:#64748B;">No image</div>',
                        unsafe_allow_html=True,
                    )

                designation_text = str(row.get("designation") or row.get("product_designation") or "")
                description_text = str(row.get("description") or row.get("product_description") or "")

                label_short = designation_text[:100] + "..." if len(designation_text) > 100 else designation_text
                st.markdown(f'<p style="font-size:0.85rem;color:#E2E8F0;margin:6px 0 2px 0">{label_short}</p>', unsafe_allow_html=True)

                with st.expander("View details"):
                    st.markdown(f"**Designation**  \n{designation_text}")
                    st.markdown(f"**Description**  \n{description_text if description_text.strip() else '*No description available*'}")
