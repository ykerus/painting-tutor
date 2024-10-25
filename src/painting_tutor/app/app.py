"""Main application entry point for the News Reader app.
Makes use of app components defined in the `components` module.
"""

import time

import streamlit as st

from painting_tutor.app import components
from painting_tutor.app.utils import check_password
from painting_tutor.logs import configure_logger
from painting_tutor.segmentation import load_model_from_bucket, load_model_from_local


@st.cache_resource()
def configure_logging():
    configure_logger()


@st.cache_resource()
def load_segmentation_model():
    st.session_state["sam"] = load_model_from_local()


def init_session_state():
    for key in ["image_rgb", "image_bgr", "sam", "sam_result", "sam_masks"]:
        if key not in st.session_state:
            st.session_state[key] = None


# if not check_password():
#     st.stop()

st.set_page_config(
    page_title="Painting Tutor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_logging()
init_session_state()
load_segmentation_model()

st.sidebar.header("📷 Image Upload")
components.upload_image(st.sidebar)
components.process_image_on_button_press(st.sidebar)
components.select_n_colors(st.sidebar)
components.select_n_masks(st.sidebar)
# if st.sidebar.button("Rerun"):
#     st.rerun()

st.subheader("🖼️ Painting Tutor")
cols = st.columns(2)
components.show_image(cols[0], st.session_state["image_rgb"])
# components.show_sam_result(cols[1])
components.show_kmeans_colors(cols[1])
