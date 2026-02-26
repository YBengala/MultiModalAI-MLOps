import streamlit as st


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #243447;
            border: 1px solid #38BDF8;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }
        .badge-correct {
            display: inline-block;
            background-color: #16A34A;
            color: #F0FDF4;
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-wrong {
            display: inline-block;
            background-color: #DC2626;
            color: #FEF2F2;
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .product-card {
            background-color: #243447;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            text-align: center;
        }
        .product-card img {
            border-radius: 6px;
            max-height: 160px;
            object-fit: contain;
        }
        .product-designation {
            font-size: 0.78rem;
            color: #CBD5E1;
            margin-top: 6px;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
