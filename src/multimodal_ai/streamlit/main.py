import streamlit as st

st.set_page_config(
    page_title="Rakuten MLOps",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/1_demo.py", title="Prediction Demo", icon="🔮"),
    st.Page("pages/2_gallery.py", title="Product Gallery", icon="🗂️"),
    st.Page("pages/3_pipeline.py", title="Pipeline & Model", icon="⚙️"),
    st.Page("pages/4_monitoring.py", title="Monitoring", icon="📊"),
]

pg = st.navigation(pages)
pg.run()
