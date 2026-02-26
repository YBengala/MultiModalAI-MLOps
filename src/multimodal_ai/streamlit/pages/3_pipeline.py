import base64
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
from components.styles import apply_custom_css

apply_custom_css()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "Rakuten_Multimodal_Fusion"

# Architecture

st.subheader("Pipeline Architecture")

st.markdown(
    "The pipeline is fully automated end-to-end. "
    "Raw CSV data is ingested by Airflow, stored in PostgreSQL and MinIO (images + DVC-versioned embeddings). "
    "Once ingestion completes, an **Airflow Dataset** signal automatically triggers the training DAG — "
    "Optuna runs 20 HPO trials, then the best model is trained with PyTorch and promoted to the "
    "MLflow registry if it beats the current Production F1-macro. "
    "FastAPI loads the Production model at startup; Streamlit calls the API for real-time predictions."
)

st.graphviz_chart("""
digraph pipeline {
    rankdir=LR
    fontname="Helvetica"
    graph [bgcolor="#1A2332" fontcolor="#94A3B8" fontsize=11]
    node  [fontname="Helvetica" fontsize=11 style=filled]
    edge  [fontname="Helvetica" fontsize=9 fontcolor="#94A3B8" color="#38BDF8" arrowsize=0.8]

    subgraph cluster_data {
        label="Data Layer"
        fontcolor="#94A3B8" color="#2D4A6A" style=dashed penwidth=1.5

        csv  [label="CSV Data\\n(Rakuten)"      shape=note      fillcolor="#1E3A5F" fontcolor="#E2E8F0" color="#38BDF8"]
        pg   [label="PostgreSQL\\n(Products)"   shape=cylinder  fillcolor="#1E3A5F" fontcolor="#E2E8F0" color="#38BDF8"]
        minio[label="MinIO\\n(Artifacts + DVC)" shape=cylinder  fillcolor="#1E3A5F" fontcolor="#E2E8F0" color="#38BDF8"]
    }

    subgraph cluster_airflow {
        label="Airflow Orchestration"
        fontcolor="#34D399" color="#34D399" style=solid penwidth=2

        subgraph cluster_ingestion {
            label="rakuten_ingestion DAG"
            fontcolor="#94A3B8" color="#2D4A6A" style=dashed penwidth=1.2

            ing  [label="Ingestion" shape=hexagon fillcolor="#243447" fontcolor="#E2E8F0" color="#38BDF8"]
        }

        subgraph cluster_training {
            label="rakuten_training DAG"
            fontcolor="#94A3B8" color="#2D4A6A" style=dashed penwidth=1.2

            tun  [label="HPO Tuning\\n(Optuna)"  shape=box fillcolor="#243447" fontcolor="#E2E8F0" color="#38BDF8"]
            trn  [label="Training\\n(PyTorch)"   shape=box fillcolor="#243447" fontcolor="#E2E8F0" color="#38BDF8"]
            trk  [label="MLflow Tracking\\n(runs + metrics)" shape=box fillcolor="#243447" fontcolor="#E2E8F0" color="#38BDF8"]
            reg  [label="MLflow Registry\\n(Production)" shape=box fillcolor="#0F2744" fontcolor="#38BDF8" color="#38BDF8" penwidth=2.5]
        }
    }

    subgraph cluster_serving {
        label="Serving Layer"
        fontcolor="#94A3B8" color="#2D4A6A" style=dashed penwidth=1.5

        api  [label="FastAPI\\nInference"  shape=box  fillcolor="#243447" fontcolor="#E2E8F0" color="#38BDF8"]
        app  [label="Streamlit\\nDemo"     shape=box  fillcolor="#0F2744" fontcolor="#38BDF8" color="#38BDF8" penwidth=2.5]
    }

    csv  -> ing  [label="raw data"]
    ing  -> pg   [label="store"]
    ing  -> minio[label="images + embeddings\\n(DVC versioned)"]
    tun  -> trn  [label="best params"]
    minio -> tun [label="Airflow Dataset\\n(auto-trigger)" style=dashed color="#F59E0B" fontcolor="#F59E0B"]
    trn  -> trk  [label="log runs"]
    trk  -> reg  [label="promote best"]
    reg  -> api  [label="load model"]
    pg   -> app  [label="gallery"  style=dashed color="#64748B"]
    minio -> app [label="images"   style=dashed color="#64748B"]
    api  -> app  [label="predict"]
}
""")


def _svg_b64(name: str) -> str:
    """Load a local SVG and return a base64 data URI."""
    icons_dir = Path(__file__).parent.parent / "static" / "icons"
    path = icons_dir / name
    try:
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/svg+xml;base64,{data}"
    except FileNotFoundError:
        return ""


def _badge(name: str, role: str, src: str, width: int = 32) -> str:
    return (
        f'<div class="metric-card" style="text-align:center;padding:12px 8px">'
        f'<img src="{src}" width="{width}" style="margin-bottom:8px"/>'
        f'<div style="font-size:0.9rem;font-weight:700;color:#E2E8F0">{name}</div>'
        f'<div style="font-size:0.7rem;color:#94A3B8;margin-top:3px">{role}</div>'
        f"</div>"
    )


st.markdown("---")
row1 = [
    ("Airflow", "Orchestration", "airflow.svg"),
    ("DVC", "Data Versioning", "dvc.svg"),
    ("MLflow", "Tracking & Registry", "mlflow.svg"),
    ("MinIO", "Artifacts, Images & DVC", "minio.svg"),
    ("PostgreSQL", "Database", "postgresql.svg"),
]
row2 = [
    ("FastAPI", "Serving", "fastapi.svg"),
    ("Streamlit", "Demo", "streamlit.svg"),
    ("Docker", "Containers", "docker.svg"),
    ("PyTorch", "Training", "pytorch.svg"),
    ("HuggingFace", "sentence-transformers + timm", "huggingface.svg"),
]

for row in (row1, row2):
    cols = st.columns(5)
    for col, (name, role, icon) in zip(cols, row):
        with col:
            st.markdown(_badge(name, role, _svg_b64(icon)), unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

st.divider()

# MLflow Dashboard

st.subheader("MLflow Dashboard")


@st.cache_data(ttl=60)
def load_production_info() -> dict | None:
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        prod_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        prod = next((v for v in prod_versions if v.current_stage == "Production"), None)
        if prod is None:
            return None
        run = client.get_run(prod.run_id)
        p = run.data.params
        duration_ms = (run.info.end_time or 0) - (run.info.start_time or 0)
        duration_min = round(duration_ms / 60000, 1) if duration_ms > 0 else None
        return {
            "version": prod.version,
            "run_id": prod.run_id,
            "f1": run.data.metrics.get("final_val_f1_macro", 0.0),
            "acc": run.data.metrics.get("final_val_acc", 0.0),
            "data_run_id": run.data.tags.get("data_run_id", "—"),
            "git_sha": run.data.tags.get("git_sha", "—"),
            "duration_min": duration_min,
            # Dataset
            "nb_samples": p.get("data_nb_samples", "—"),
            "nb_classes": p.get("data_nb_classes", "—"),
            "nb_train": p.get("data_nb_train", "—"),
            "nb_val": p.get("data_nb_val", "—"),
            "imbalance": p.get("data_imbalance_ratio", "—"),
            # Hyperparams
            "lr": p.get("learning_rate", "—"),
            "batch_size": p.get("batch_size", "—"),
            "dropout": round(float(p["dropout"]), 4) if "dropout" in p else "—",
            "weight_decay": p.get("weight_decay", "—"),
            "epochs": p.get("epochs", "—"),
            "activation": p.get("activation", "—"),
            "hidden_l1": p.get("hidden_l1", "—"),
            "hidden_l2": p.get("hidden_l2", "—"),
            "hidden_l3": p.get("hidden_l3", "—"),
            "device": run.data.tags.get("device", "—").upper(),
            "trained_at": (
                datetime.fromtimestamp(
                    run.info.end_time / 1000, tz=ZoneInfo("Europe/Paris")
                ).strftime("%Y-%m-%d %H:%M")
                if run.info.end_time
                else "—"
            ),
        }
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_all_versions() -> list[dict]:
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        return [
            {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
            for v in versions
        ]
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_runs_df() -> pd.DataFrame:
    try:
        return mlflow.search_runs(
            experiment_names=["Rakuten_Multimodal_Training"],
            tracking_uri=MLFLOW_TRACKING_URI,
            order_by=["start_time ASC"],
        )
    except Exception:
        return pd.DataFrame()


prod_info = load_production_info()

if prod_info is None:
    st.warning("Cannot reach MLflow or no model in Production.")
else:
    # Production metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Production Version", f"v{prod_info['version']}")
    c2.metric("F1-macro", f"{prod_info['f1']:.4f}")
    c3.metric("Accuracy", f"{prod_info['acc']:.4f}")
    c4.metric(
        "Training Duration",
        f"{prod_info['duration_min']} min" if prod_info["duration_min"] else "—",
    )
    c5.metric("Device", prod_info["device"])
    c6.metric("Data Batch", prod_info["data_run_id"])

    st.caption(f"Trained at : {prod_info['trained_at']} (Paris) · git {prod_info['git_sha']}")

    st.divider()

    # Dataset + Hyperparams
    col_data, col_hp = st.columns(2)

    with col_data:
        st.markdown("**Dataset**")
        st.markdown(
            f'<div class="metric-card">'
            f'<table style="width:100%;font-size:0.85rem;border-collapse:collapse">'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Total samples</td><td style="color:#E2E8F0;text-align:right">{prod_info["nb_samples"]}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Classes</td><td style="color:#E2E8F0;text-align:right">{prod_info["nb_classes"]}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Train / Val</td><td style="color:#E2E8F0;text-align:right">{prod_info["nb_train"]} / {prod_info["nb_val"]}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Imbalance ratio</td><td style="color:#E2E8F0;text-align:right">{prod_info["imbalance"]}×</td></tr>'
            f"</table></div>",
            unsafe_allow_html=True,
        )

    with col_hp:
        st.markdown("**Best Hyperparameters (Optuna)**")
        st.markdown(
            f'<div class="metric-card">'
            f'<table style="width:100%;font-size:0.85rem;border-collapse:collapse">'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Learning rate</td><td style="color:#38BDF8;text-align:right;font-weight:600">{prod_info["lr"]}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Batch size</td><td style="color:#38BDF8;text-align:right;font-weight:600">{prod_info["batch_size"]}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Dropout</td><td style="color:#38BDF8;text-align:right;font-weight:600">{prod_info["dropout"]}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Weight decay</td><td style="color:#38BDF8;text-align:right;font-weight:600">{prod_info["weight_decay"]}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Epochs</td><td style="color:#38BDF8;text-align:right;font-weight:600">{prod_info["epochs"]}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Activation</td><td style="color:#38BDF8;text-align:right;font-weight:600">{prod_info["activation"]}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:4px 0">Hidden layers</td><td style="color:#38BDF8;text-align:right;font-weight:600">{prod_info["hidden_l1"]} → {prod_info["hidden_l2"]} → {prod_info["hidden_l3"]}</td></tr>'
            f"</table></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # F1-macro evolution chart
    runs = load_runs_df()

    if not runs.empty:
        metric_cols = [c for c in runs.columns if "final_val_f1_macro" in c]
        acc_cols = [c for c in runs.columns if "final_val_acc" in c]

        if metric_cols:
            f1_col = metric_cols[0]
            acc_col = acc_cols[0] if acc_cols else None

            df_plot = runs[["run_id", f1_col]].dropna(subset=[f1_col]).copy()
            df_plot["run"] = [f"run {i + 1}" for i in range(len(df_plot))]

            fig = px.line(
                df_plot,
                x="run",
                y=f1_col,
                markers=True,
                title="F1-macro across training runs",
                color_discrete_sequence=["#38BDF8"],
                labels={f1_col: "F1-macro", "run": ""},
            )
            fig.update_layout(
                plot_bgcolor="#1A2332",
                paper_bgcolor="#1A2332",
                font_color="#E2E8F0",
                margin={"t": 50, "b": 20},
            )
            fig.add_hline(
                y=prod_info["f1"],
                line_dash="dot",
                line_color="#F59E0B",
                annotation_text="Current Production",
                annotation_font_color="#F59E0B",
            )
            st.plotly_chart(fig, width='stretch')

            # Runs table with status badge
            all_versions = load_all_versions()
            stage_map = {v["run_id"]: v["stage"] for v in all_versions}

            display_cols = ["run_id", f1_col]
            if acc_col:
                display_cols.append(acc_col)
            display_cols += ["start_time"]
            tag_cols = [c for c in runs.columns if "tags.data_run_id" in c]
            display_cols += tag_cols

            rename_map = {f1_col: "F1-macro"}
            if acc_col:
                rename_map[acc_col] = "Accuracy"

            # duration computed before sort so index alignment is correct
            runs_with_dur = runs.copy()
            if "start_time" in runs.columns and "end_time" in runs.columns:
                runs_with_dur["Duration (min)"] = (
                    (runs["end_time"] - runs["start_time"]) / 60000
                ).round(1)
                display_cols.append("Duration (min)")

            df_table = (
                runs_with_dur[display_cols]
                .rename(columns=rename_map)
                .sort_values("start_time", ascending=False)
                .copy()
            )

            # format start_time as readable Paris time
            if "start_time" in df_table.columns:
                df_table["start_time"] = pd.to_datetime(df_table["start_time"], utc=True).dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d %H:%M")

            # status badge
            def make_badge(run_id):
                stage = stage_map.get(run_id, "")
                if stage == "Production":
                    return "🟢 Production"
                elif stage == "Archived":
                    return "⚫ Archived"
                else:
                    return "⚪ —"

            df_table["Status"] = df_table["run_id"].map(make_badge)

            st.dataframe(df_table, width='stretch')
