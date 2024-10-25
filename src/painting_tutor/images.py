import cv2
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from sklearn.cluster import KMeans


def smooth_image_mean_shift(image_rgb, spatial_radius=21, color_radius=101):
    return cv2.pyrMeanShiftFiltering(image_rgb, spatial_radius, color_radius)


def smooth_image_gaussian(image_rgb, sigma=1):
    ksize = sigma * 5 if sigma % 2 == 1 else sigma * 5 + 1
    return cv2.GaussianBlur(image_rgb, (ksize, ksize), sigma)


def smooth_image_segments_gaussian(image, masks, sigma=1):
    smoothed_image = image.copy().astype(float) / 255
    mask_covered = np.zeros_like(image, dtype=bool)
    for mask in masks:
        mask_covered[mask] = True
        mask_cut = cut_out_mask(image, mask)
        mask_cut_smoothed = smooth_mask_cut(mask_cut, sigma)
        smoothed_image[mask] = mask_cut_smoothed[mask]

    mask_cut = cut_out_mask(image, ~mask_covered)
    mask_cut_smoothed = smooth_mask_cut(mask_cut, sigma)
    smoothed_image[~mask_covered] = mask_cut_smoothed[~mask_covered]

    return (smoothed_image * 255).astype(np.uint8)


def fill_nans_with_nearest_values(image):
    mask = np.where(~np.isnan(image))
    interp = NearestNDInterpolator(np.transpose(mask), image[mask])
    filled_image = interp(*np.indices(image.shape))
    return filled_image


def fill_nans_with_mean_values(image):
    mean_rgb = np.nanmean(image, axis=(0, 1))
    image_filled = image.copy()
    mask = np.isnan(image).any(axis=2)
    image_filled[mask] = mean_rgb
    return image_filled


def get_kmeans_pixels(mask, image, n_colors=1):
    img = image.copy()
    pixels = img[mask]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    pixels_kmeans = kmeans.cluster_centers_[kmeans.labels_].astype(int)
    return pixels_kmeans


def create_kmeans_image(masks, image, n_colors=1):
    kmeans_image = image.copy()
    mask_covered = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        mask_covered[mask] = True
        pixels_kmeans = get_kmeans_pixels(mask, image, n_colors)
        kmeans_image[mask] = pixels_kmeans

    mask = ~mask_covered
    pixels_kmeans = get_kmeans_pixels(mask, image, n_colors)
    kmeans_image[mask] = pixels_kmeans

    return kmeans_image


def cut_out_mask(image, mask):
    img = image.copy().astype(float) / 255
    img[~mask] = np.nan
    return img


def smooth_mask_cut(mask_cut, sigma=1):
    nan_mask = np.isnan(mask_cut)
    mask_cut_filled = fill_nans_with_mean_values(mask_cut)
    mask_cut_smoothed = smooth_image_gaussian(mask_cut_filled, sigma)
    mask_cut_smoothed[nan_mask] = np.nan
    return mask_cut_smoothed


def make_black_and_white(image):
    return cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
