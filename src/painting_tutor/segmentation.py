import supervision as sv
from segment_anything import SamAutomaticMaskGenerator


def segment_image(image, mask_generator: SamAutomaticMaskGenerator):
    sam_result = mask_generator.generate(image)
    return sam_result


def get_masks(sam_result):
    masks = [
        mask["segmentation"] for mask in sorted(sam_result, key=lambda x: x["area"], reverse=True)
    ]
    return masks


def get_annotated_image(sam_result, image):
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    return annotated_image
