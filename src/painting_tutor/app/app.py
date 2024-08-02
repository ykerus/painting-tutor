"""Main application entry point for the News Reader app.
Makes use of app components defined in the `components` module.
"""

import time

import streamlit as st

from painting_tutor.app import components
from painting_tutor.app.utils import check_password
from painting_tutor.logs import configure_logger


@st.cache_resource()
def configure_logging():
    configure_logger()


def init_session_state():
    for key in ["image"]:
        if key not in st.session_state:
            st.session_state[key] = None


if not check_password():
    st.stop()

st.set_page_config(
    page_title="Painting Tutor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_logging()
init_session_state()

st.sidebar.header("📷 Image Upload")
image = components.upload_image(st.sidebar)

st.subheader("🖼️ Painting Tutor")
cols = st.columns(2)
components.show_image(cols[0], image)
