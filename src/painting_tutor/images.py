from sklearn.cluster import KMeans


def get_kmeans_pixels(mask, image, n_colors=1):
    img = image.copy()
    pixels = img[mask]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    pixels_kmeans = kmeans.cluster_centers_[kmeans.labels_].astype(int)
    return pixels_kmeans


def create_kmeans_image(masks, image, n_colors=1, n_masks=5):
    kmeans_image = image.copy()
    for mask in masks[:n_masks]:
        pixels_kmeans = get_kmeans_pixels(mask, image, n_colors)
        kmeans_image[mask] = pixels_kmeans

    return kmeans_image
