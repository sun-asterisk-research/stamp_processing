import logging
from typing import Tuple, Union

import gdown
import numpy as np
import numpy.typing as npt
import torch
from torch import device as torch_device
from torch.nn import Module

from stamp_processing.module.yolov5 import YOLO_DIR


logging.basicConfig(format="%(levelname)s - %(message)s'")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DETECTOR_WEIGHT_ID = "1YHH7pLoZEdyxw2AoLz9G4lrq6uuxweYB"
REMOVER_WEIGHT_ID = "1Hd79M8DhCwjFuT198R-QB7ozQbHRGcGM"


def select_device(device: str = "") -> torch_device:
    """Return a torch.device instance"""
    cpu = device.lower() == "cpu"
    cuda = not cpu and torch.cuda.is_available()
    return torch_device("cuda:0" if cuda else "cpu")


def load_yolo_model(weight_path: str, device: str) -> Tuple[Module, int]:
    """Load yolov5 model from specified path using torch hub"""
    model = torch.hub.load(str(YOLO_DIR), "custom", path=weight_path, source="local", force_reload=True)
    model.to(device)
    return model, model.stride


def download_weight(file_id: str, output: Union[str, None] = None, quiet: bool = False) -> None:
    """Download model weight from Google Drive given the file ID"""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.cached_download(url=url, path=output, quiet=quiet)
    except Exception as e:
        logger.error(e)
        logger.error("Something went wrong when downloading the weight")
        logger.error(
            "Check your internet connection or manually download the weight "
            f"at https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        )


def check_image_shape(image: npt.NDArray) -> None:
    """Check if input image is valid"""
    if not isinstance(image, np.ndarray):
        raise TypeError("Invalid Type: List value must be of type np.ndarray")
    else:
        if len(image.shape) != 3:
            raise ValueError("Invalid image shape")
        if image.shape[-1] != 3:
            raise ValueError("Image must be 3 dimensional")
