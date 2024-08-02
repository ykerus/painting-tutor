import logging
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

logger = logging.getLogger(__name__)


def upload_image(position: DeltaGenerator) -> None:
    image = None
    uploaded_file = position.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    st.session_state["image"] = image


def show_image(position: DeltaGenerator, image: Optional[np.ndarray]) -> None:
    if image is None:
        position.info("_No image uploaded yet._")
    else:
        position.image(image, use_column_width=True)
