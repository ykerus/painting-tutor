from typing import Optional

import cv2
import numpy as np


def extract_major_lines(
    image: np.ndarray,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
    min_line_length: int = 50,
) -> np.ndarray:
    if low_threshold is None or high_threshold is None:
        v = np.median(image)
        sigma = 0.33  # You can adjust this value as needed
        low_threshold = int(max(0, (1.0 - sigma) * v))
        high_threshold = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(image, low_threshold, high_threshold)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, False) > min_line_length]

    line_mask = np.zeros_like(image)
    cv2.drawContours(line_mask, filtered_contours, -1, [255] * 3, thickness=8)

    return line_mask.astype(bool)


def lines_to_image(line_mask):
    return (~line_mask).astype(np.uint8) * 255


def overlay_lines(image, line_mask):
    overlay = image.copy()
    overlay[line_mask] = 0
    return overlay
