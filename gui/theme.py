import streamlit as st


def configure_page() -> None:
    """设置页面基础信息。"""
    st.set_page_config(
        page_title="房屋布局强化学习工作台",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme() -> None:
    """注入简洁的科研风格主题。"""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
            color: #16324f;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }
        h1, h2, h3 {
            color: #16324f;
            letter-spacing: 0.02em;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d9e2ec;
        }
        [data-testid="stMetricValue"] {
            color: #16324f;
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #c8d5e3;
            background: #ffffff;
            color: #16324f;
        }
        .stButton > button:hover {
            border-color: #4c78a8;
            color: #0b5fa5;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            background: rgba(255, 255, 255, 0.7);
            padding: 10px 16px;
        }
        .stInfo, .stSuccess, .stWarning, .stError {
            border-radius: 8px;
        }
        /* 监控 fragment 刷新时，禁止 Streamlit 把旧内容淡白处理 */
        [data-stale="true"] {
            opacity: 1 !important;
            filter: none !important;
        }
        /* 隐藏刷新时可能短暂出现的骨架占位，减少“整块发白”感 */
        [data-testid="stSkeleton"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
