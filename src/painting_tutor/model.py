import os

import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def load_model_from_local() -> SamAutomaticMaskGenerator:
    model_name = "sam_vit_h"
    model_dir = "model"
    model_type = model_name.split("sam_")[1] # vit_...
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_fname = [path for path in os.listdir(model_dir) if model_name in path][0]
    model_path = os.path.join(model_dir, model_fname)

    sam = sam_model_registry[model_type](checkpoint=model_path).to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator