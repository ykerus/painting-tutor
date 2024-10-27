import logging

import cv2
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from painting_tutor.images import create_kmeans_image, get_kmeans_pixels, make_black_and_white
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
        st.session_state["image_name"] = uploaded_file.name
        st.session_state["image_size"] = uploaded_file.size


def process_image_button(position: DeltaGenerator) -> bool:
    if st.session_state["image_rgb"] is None:
        return False
    return position.button("Process Image", key="process_image")


def select_n_colors(position: DeltaGenerator) -> None:
    n_colors = position.slider("Number of colors", min_value=1, max_value=10, value=3)
    st.session_state["n_colors"] = n_colors


def select_n_masks(position: DeltaGenerator) -> None:
    if st.session_state["sam_masks"] is not None:
        max_masks = len(st.session_state["sam_masks"])
    else:
        max_masks = 30

    n_colors = position.slider(
        "Number of masks", min_value=1, max_value=max_masks, value=min(30, max_masks)
    )
    st.session_state["n_masks"] = n_colors


def select_min_line_length(position: DeltaGenerator) -> None:
    min_line_length = position.slider("Min. line length", min_value=0, max_value=500, value=100)
    st.session_state["min_line_length"] = min_line_length


def select_line_min_threshold(position: DeltaGenerator) -> None:
    line_min_threshold = position.slider(
        "Line min. threshold", min_value=0, max_value=500, value=50
    )
    st.session_state["line_min_threshold"] = line_min_threshold


def select_line_max_threshold(position: DeltaGenerator) -> None:
    line_max_threshold = position.slider(
        "Line max. threshold", min_value=0, max_value=500, value=150
    )
    st.session_state["line_max_threshold"] = line_max_threshold


def select_sigma(position: DeltaGenerator) -> None:
    sigma = position.slider("Sigma", min_value=1, max_value=15, value=5)
    st.session_state["sigma"] = sigma


def checkbox_black_and_white(position: DeltaGenerator) -> None:
    black_and_white = position.checkbox("Black and white", value=False)
    st.session_state["black_and_white"] = black_and_white


def checkbox_mask_only(position: DeltaGenerator) -> None:
    mask_only = position.checkbox("Show mask only", value=False)
    st.session_state["mask_only"] = mask_only


def checkbox_mask_background_black(position: DeltaGenerator) -> None:
    mask_background_black = position.checkbox("Dark mask background", value=False)
    st.session_state["mask_background_black"] = mask_background_black


def select_mask_index(position: DeltaGenerator) -> None:
    if st.session_state["sam_masks"] is None:
        max_mask_index = 15
    else:
        max_mask_index = len(st.session_state["sam_masks"])

    mask_index = position.slider(
        "Mask index",
        min_value=0,
        max_value=min(15, max_mask_index),
        value=0,
    )
    st.session_state["mask_index"] = mask_index


def checkbox_cool_mask(position: DeltaGenerator) -> None:
    show_cool_mask = position.checkbox("Show cool mask", value=False)
    st.session_state["show_cool_mask"] = show_cool_mask


def show_image(position: DeltaGenerator, image, ignore_settings=False) -> None:
    if image is None:
        return

    if st.session_state["black_and_white"] and not ignore_settings:
        image = make_black_and_white(image)

    if st.session_state["mask_only"] and not ignore_settings:
        mask = st.session_state["sam_masks"][st.session_state["mask_index"]]
        image = image.copy()

        if st.session_state["mask_background_black"]:
            fill_value = 0
        else:
            fill_value = 255 if np.max(image) > 1 else 1

        image[~mask] = fill_value

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

    kmeans_image = create_kmeans_image(
        st.session_state["masks"],
        st.session_state["image_rgb"],
        n_colors=n_colors,
    )
    show_image(position, kmeans_image)
