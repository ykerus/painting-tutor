import logging
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from painting_tutor.images import create_kmeans_image, get_kmeans_pixels
from painting_tutor.segmentation import get_annotated_image, get_masks, segment_image

logger = logging.getLogger(__name__)


def upload_image(position: DeltaGenerator) -> None:
    image = None
    uploaded_file = position.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.session_state["image_bgr"] = image
        st.session_state["image_rgb"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def process_image_on_button_press(position: DeltaGenerator) -> None:
    if st.session_state["image_rgb"] is None:
        return

    if position.button("Process Image", key="process_image"):
        sam_result = segment_image(st.session_state["image_rgb"], st.session_state["sam"])
        st.session_state["sam_result"] = sam_result
        st.session_state["sam_masks"] = get_masks(sam_result)


def select_n_colors(position: DeltaGenerator) -> None:
    n_colors = position.slider("Number of colors", min_value=1, max_value=10, value=1)
    st.session_state["n_colors"] = n_colors


def select_n_masks(position: DeltaGenerator) -> None:
    if st.session_state["sam_masks"] is not None:
        max_masks = len(st.session_state["sam_masks"])
    else:
        max_masks = 10

    n_colors = position.slider("Number of masks", min_value=1, max_value=max_masks, value=max_masks)
    st.session_state["n_masks"] = n_colors


def show_image(position: DeltaGenerator, image) -> None:
    if image is not None:
        position.image(image, use_column_width=True)


def show_sam_result(position: DeltaGenerator) -> None:
    sam_result = st.session_state["sam_result"]

    if sam_result is not None:
        annotated_image = get_annotated_image(sam_result, st.session_state["image_rgb"])
        show_image(position, annotated_image)


def show_kmeans_colors(position: DeltaGenerator) -> None:

    if st.session_state["sam_masks"] is None:
        return

    n_colors = st.session_state["n_colors"]
    n_masks = st.session_state["n_masks"]

    kmeans_image = create_kmeans_image(
        st.session_state["sam_masks"],
        st.session_state["image_rgb"],
        n_colors=n_colors,
        n_masks=n_masks,
    )
    show_image(position, kmeans_image)
