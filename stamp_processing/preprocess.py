from typing import List, Set, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch


def create_batch(
    images: npt.NDArray, shapes: Set[Tuple[int, int]], batch_size: int = 16
) -> Tuple[List[List[npt.NDArray]], List[int]]:
    """
    - Input:
        +) images: List images
        +) shapes: set of all shapes of input images
        +) batch_size: number image in one batch
    - Output:
        +) images_batch: batch of images for inference
        +) indices: order of all input images
    """
    split_batch = []
    images_batch = []
    for shape in shapes:
        mini_batch = []
        images_mini_batch = []  # type: ignore
        for idx, img in enumerate(images):
            if img.shape == shape:
                mini_batch.append(idx)
                if len(images_mini_batch) < batch_size:
                    images_mini_batch.append(img)
                else:
                    images_batch.append(images_mini_batch)
                    images_mini_batch = []
                    images_mini_batch.append(img)
        images_batch.append(images_mini_batch)
        split_batch.append(mini_batch)
    del images_mini_batch

    indices = [item for sublist in split_batch for item in sublist]
    return images_batch, indices


def process_image(img: npt.NDArray, device: str = "cpu") -> npt.NDArray:
    """Preprocess function for yolov5

    Args:
        img (npt.NDArray): Input image
        device (str, optional): torch device. Defaults to "cpu".

    Returns:
        npt.NDArray: preprocessed image
    """
    height, width = img.shape[:2]
    top = (640 - height) // 2
    bottom = 640 - height - top
    left = (640 - width) // 2
    right = 640 - width - left
    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0
    return img
