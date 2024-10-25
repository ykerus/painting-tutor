import pickle

import streamlit as st

from painting_tutor.edges import extract_major_lines, lines_to_image, overlay_lines
from painting_tutor.images import create_kmeans_image, smooth_image_segments_gaussian
from painting_tutor.segmentation import get_masks, segment_image


@st.cache_data
def segment_image_cached(image):
    output = segment_image(image, st.session_state["sam"])
    pickle.dump(output, open("segment_image_2.pkl", "wb"))

    # output = pickle.load(open("segment_image.pkl", "rb"))
    return output


@st.cache_data
def smooth_image_cached(image, masks, sigma):
    return smooth_image_segments_gaussian(image, masks, sigma)


@st.cache_data
def extract_lines_cached(image, low_threshold, high_threshold, min_line_length):
    return extract_major_lines(
        image=image,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        min_line_length=min_line_length,
    )


@st.cache_data
def create_kmeans_image_cached(masks, image, n_colors):
    return create_kmeans_image(
        masks,
        image,
        n_colors=n_colors,
    )


def process_image():
    st.session_state["sam_result"] = segment_image_cached(st.session_state["image_rgb"])
    st.session_state["sam_masks"] = get_masks(st.session_state["sam_result"])
    st.session_state["masks"] = st.session_state["sam_masks"][: st.session_state["n_masks"]]
    st.session_state["image_smoothed"] = smooth_image_cached(
        image=st.session_state["image_rgb"],
        masks=st.session_state["masks"],
        sigma=st.session_state["sigma"],
    )
    st.session_state["line_mask"] = extract_lines_cached(
        image=st.session_state["image_smoothed"],
        low_threshold=st.session_state["line_min_threshold"],
        high_threshold=st.session_state["line_max_threshold"],
        min_line_length=st.session_state["min_line_length"],
    )
    st.session_state["line_image"] = lines_to_image(st.session_state["line_mask"])
    st.session_state["line_overlay"] = overlay_lines(
        image=st.session_state["image_smoothed"],
        line_mask=st.session_state["line_mask"],
    )

    st.session_state["means_image"] = create_kmeans_image_cached(
        masks=st.session_state["masks"],
        image=st.session_state["image_rgb"],
        n_colors=1,
    )
    st.session_state["kmeans_image"] = create_kmeans_image_cached(
        masks=st.session_state["masks"],
        image=st.session_state["image_rgb"],
        n_colors=st.session_state["n_colors"],
    )

    st.session_state["kmeans_image_smoothed"] = create_kmeans_image_cached(
        masks=st.session_state["masks"],
        image=st.session_state["image_smoothed"],
        n_colors=st.session_state["n_colors"],
    )

    st.session_state["mask"] = st.session_state["sam_masks"][st.session_state["mask_index"]]
