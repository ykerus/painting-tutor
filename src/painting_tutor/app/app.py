"""Main application entry point for the News Reader app.
Makes use of app components defined in the `components` module.
"""

import streamlit as st

from painting_tutor.app import components
from painting_tutor.app.process import process_image
from painting_tutor.logs import configure_logger
from painting_tutor.model import load_model_from_local


@st.cache_resource()
def configure_logging():
    configure_logger()


@st.cache_resource()
def load_segmentation_model():
    return load_model_from_local()


def init_session_state():
    if "first_run" not in st.session_state:
        st.session_state["first_run"] = True
        st.session_state["image_rgb"] = None
        st.session_state["sam_masks"] = None


st.set_page_config(
    page_title="Painting Tutor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_logging()
init_session_state()
st.session_state["sam"] = load_segmentation_model()

st.sidebar.header("📷 Image Upload")
components.upload_image(st.sidebar)

components.select_line_x(st.sidebar)
components.select_line_y(st.sidebar)


components.select_n_colors(st.sidebar)
components.select_n_masks(st.sidebar)

components.select_min_line_length(st.sidebar)
components.select_line_min_threshold(st.sidebar)
components.select_line_max_threshold(st.sidebar)

components.select_sigma(st.sidebar)

components.checkbox_black_and_white(st.sidebar)
components.checkbox_mask_only(st.sidebar)
components.checkbox_mask_background_black(st.sidebar)
components.select_mask_index(st.sidebar)

components.checkbox_cool_mask(st.sidebar)

st.subheader("🖼️ Painting Tutor")

if st.session_state["image_rgb"] is None:
    st.info("Please upload an image.")
    st.stop()

process_image()
if st.session_state["first_run"]:
    st.session_state["first_run"] = False
    st.rerun()

cols = st.columns(2)

components.show_image(cols[0], st.session_state["image_rgb"])
components.show_image(cols[1], st.session_state["line_overlay"])

components.show_image(cols[0], st.session_state["means_image"])
components.show_image(cols[1], st.session_state["line_image"])

components.show_image(cols[0], st.session_state["image_rgb"])
components.show_image(cols[1], st.session_state["image_smoothed"])

components.show_image(cols[0], st.session_state["kmeans_image_smoothed"])
components.show_image(cols[1], st.session_state["kmeans_image"])

if st.session_state["show_cool_mask"]:
    components.show_image(cols[0], st.session_state["cool_mask"], ignore_settings=True)
