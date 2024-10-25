import os
from typing import Literal

import logging

import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

logger = logging.getLogger(__name__)


def load_model_from_local(
    size: Literal["big", "large", "huge"] = "huge", model_dir="model"
) -> SamAutomaticMaskGenerator:
    
    logger.info(f"Loading '{size}' model from '/{model_dir}'")
    
    model_name = f"sam_vit_{size[0]}"
    model_type = model_name.split("sam_")[1] # vit_...
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_fname = [path for path in os.listdir(model_dir) if model_name in path][0]
    model_path = os.path.join(model_dir, model_fname)

    sam = sam_model_registry[model_type](checkpoint=model_path).to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator