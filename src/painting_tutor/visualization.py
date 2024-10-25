from matplotlib import pyplot as plt
import supervision as sv

from painting_tutor.images import make_black_and_white



def show_image(image, show=True, black_and_white=False, mask=None, use_mask=False):
    if black_and_white:
        image = make_black_and_white(image)
        
    if mask is not None and use_mask:
        image = image.copy()
        image[~mask] = 255
         
    plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    if show:
        plt.show()

def visualize_segments(sam_result, image_bgr):
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections.from_sam(sam_result=sam_result)

    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    sv.plot_images_grid(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=["source image", "segmented image"],
    )
