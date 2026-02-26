import io
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import requests
import streamlit as st
from components.minio_client import fetch_image
from components.styles import apply_custom_css
from PIL import Image

apply_custom_css()

st.markdown(
    '<h1 style="font-size:3rem;font-weight:800;margin-bottom:0">Rakuten MLOps — Multimodal Classification</h1>',
    unsafe_allow_html=True,
)

INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8000")
DB_HOST = os.getenv("RAKUTEN_DB_HOST", "localhost")
DB_PORT = os.getenv("RAKUTEN_DB_PORT", "5432")
DB_NAME = os.getenv("RAKUTEN_DB_NAME", "rakuten_db")
DB_USER = os.getenv("RAKUTEN_DB_USER", "")
DB_PASSWORD = os.getenv("RAKUTEN_DB_PASSWORD", "")


def get_db_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=int(DB_PORT),
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def save_prediction(
    designation, description, predicted_class, confidence, all_scores, has_image
):
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions_prod
                    (designation, description, predicted_class, confidence, all_scores, has_image)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    designation,
                    description,
                    predicted_class,
                    confidence,
                    json.dumps(all_scores),
                    has_image,
                ),
            )
        conn.commit()
        conn.close()
    except Exception:
        pass


def load_similar_products(predicted_label: str) -> pd.DataFrame:
    try:
        conn = get_db_conn()
        df = pd.read_sql(
            "SELECT * FROM products_latest WHERE prodtype = %s LIMIT 6",
            conn,
            params=(predicted_label,),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


st.markdown(
    "End-to-end MLOps pipeline for **multimodal e-commerce product classification** — "
    "text (sentence-transformers) + image (EfficientViT) embeddings fused by a PyTorch MLP, "
    "trained on the Rakuten France catalogue, orchestrated by Airflow and tracked with MLflow."
)
st.header("🔮 Prediction Demo")
st.markdown(
    "Upload a product image and enter its designation to get an instant category prediction "
    "from the multimodal model (text + image). The model covers **27 product classes** and "
    "returns a confidence score along with the top-5 candidates."
)

with st.form("prediction_form"):
    designation = st.text_input(
        "Designation *", placeholder="Ex: Set of 6 crystal wine glasses"
    )
    description = st.text_area(
        "Description (optional)",
        placeholder="Detailed product description...",
        height=100,
    )
    image_file = st.file_uploader(
        "Product image (required)", type=["jpg", "jpeg", "png"]
    )
    save_to_db = st.checkbox("Save prediction", value=True)
    submitted = st.form_submit_button("Classify", use_container_width=True)

if submitted:
    if not designation.strip():
        st.error("Designation is required.")
    elif image_file is None:
        st.error("An image is required for classification.")
    else:
        with st.spinner("Calling inference API..."):
            try:
                data = {"designation": designation}
                if description.strip():
                    data["description"] = description

                files = {
                    "image": (image_file.name, image_file.getvalue(), image_file.type)
                }

                response = requests.post(
                    f"{INFERENCE_API_URL}/predict",
                    data=data,
                    files=files,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

                predicted_label = result.get("predicted_label", "—")
                confidence = result.get("confidence", 0.0)
                top5 = result.get("top5", [])

                # Result
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(
                        Image.open(io.BytesIO(image_file.getvalue())),
                        use_container_width=True,
                    )

                with col2:
                    st.metric("Predicted class", predicted_label)

                    st.metric("Confidence", f"{confidence:.1%}")

                    # Certitude badge
                    if confidence >= 0.75:
                        badge = '<span style="background:#16A34A;color:#F0FDF4;border-radius:6px;padding:3px 12px;font-size:0.85rem;font-weight:600;">High confidence</span>'
                    elif confidence >= 0.5:
                        badge = '<span style="background:#D97706;color:#FFF;border-radius:6px;padding:3px 12px;font-size:0.85rem;font-weight:600;">Moderate confidence</span>'
                    else:
                        badge = '<span style="background:#DC2626;color:#FEF2F2;border-radius:6px;padding:3px 12px;font-size:0.85rem;font-weight:600;">Low confidence</span>'
                    st.markdown(badge, unsafe_allow_html=True)

                    # Confidence gauge
                    gauge_color = (
                        "#16A34A"
                        if confidence >= 0.75
                        else "#D97706"
                        if confidence >= 0.5
                        else "#DC2626"
                    )
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=round(confidence * 100, 1),
                            number={
                                "suffix": "%",
                                "font": {"size": 28, "color": "#E2E8F0"},
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "tickcolor": "#64748B",
                                    "tickfont": {"color": "#64748B"},
                                },
                                "bar": {"color": gauge_color},
                                "bgcolor": "#243447",
                                "bordercolor": "#243447",
                                "steps": [
                                    {"range": [0, 50], "color": "#1A2332"},
                                    {"range": [50, 75], "color": "#1A2332"},
                                    {"range": [75, 100], "color": "#1A2332"},
                                ],
                                "threshold": {
                                    "line": {"color": "#38BDF8", "width": 2},
                                    "thickness": 0.75,
                                    "value": confidence * 100,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(
                        paper_bgcolor="#1A2332",
                        font_color="#E2E8F0",
                        height=200,
                        margin={"t": 20, "b": 0, "l": 20, "r": 20},
                    )
                    st.plotly_chart(fig_gauge, width='stretch')

                # Top-5 bar chart
                if top5:
                    df_scores = pd.DataFrame(
                        [
                            {"Class": item["label"], "Score": item["confidence"]}
                            for item in top5
                        ]
                    )
                    fig = px.bar(
                        df_scores,
                        x="Score",
                        y="Class",
                        orientation="h",
                        color_discrete_sequence=["#38BDF8"],
                        title="Top-5 predictions",
                    )
                    fig.update_layout(
                        plot_bgcolor="#1A2332",
                        paper_bgcolor="#1A2332",
                        font_color="#E2E8F0",
                        yaxis={"categoryorder": "total ascending"},
                        margin={"t": 40, "b": 20},
                    )
                    st.plotly_chart(fig, width='stretch')

                # Similar products
                similar_df = load_similar_products(predicted_label)
                if not similar_df.empty:
                    st.divider()
                    st.subheader(
                        f"🔍 Similar products in catalogue — *{predicted_label}*"
                    )
                    sim_cols = st.columns(3)
                    for idx, (_, row) in enumerate(similar_df.iterrows()):
                        with sim_cols[idx % 3]:
                            img = None
                            path = row.get("path_image_minio")
                            exists = row.get("image_exists")
                            if path and exists:
                                img = fetch_image(str(path))

                            if img:
                                st.image(img, width=200)
                            else:
                                st.markdown(
                                    '<div style="height:100px;background:#243447;border-radius:6px;display:flex;'
                                    'align-items:center;justify-content:center;color:#64748B;">No image</div>',
                                    unsafe_allow_html=True,
                                )

                            designation_text = str(
                                row.get("designation")
                                or row.get("product_designation")
                                or ""
                            )
                            label_short = (
                                designation_text[:80] + "..."
                                if len(designation_text) > 80
                                else designation_text
                            )
                            st.markdown(
                                f'<p style="font-size:0.78rem;color:#CBD5E1;margin:4px 0">{label_short}</p>',
                                unsafe_allow_html=True,
                            )

                # Save to DB
                if save_to_db:
                    all_scores = {item["label"]: item["confidence"] for item in top5}
                    save_prediction(
                        designation,
                        description or None,
                        predicted_label,
                        confidence,
                        all_scores,
                        has_image=True,
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the inference API. Make sure the service is running."
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
