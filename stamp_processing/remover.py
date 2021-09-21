import os
from typing import List, Union

import numpy as np
import numpy.typing as npt

from stamp_processing.detector import StampDetector
from stamp_processing.module.unet import *
from stamp_processing.preprocess import create_batch
from stamp_processing.utils import REMOVER_WEIGHT_ID, check_image_shape, download_weight, logger


class StampRemover:
    def __init__(
        self, detection_weight: Union[str, None] = None, removal_weight: Union[str, None] = None, device: str = "cpu"
    ):
        """Create an object to remove stamps from document images"""

        assert device == "cpu", "Currently only support cpu inference"

        if removal_weight is None:
            if not os.path.exists("tmp/"):
                os.makedirs("tmp/", exist_ok=True)
            removal_weight = os.path.join("tmp", "stamp_remover.pkl")

            logger.info("Downloading stamp remover weight from google drive")
            download_weight(REMOVER_WEIGHT_ID, output=removal_weight)
            logger.info(f"Finished downloading. Weight is saved at {removal_weight}")

        try:
            self.remover = UnetInference(removal_weight)  # type: ignore
        except Exception as e:
            logger.error(e)
            logger.error("There is something wrong when loading remover weight")
            logger.error(
                "Please make sure you provide the correct path to the weight"
                "or mannually download the weight at"
                f"https://drive.google.com/file/d/{REMOVER_WEIGHT_ID}/view?usp=sharing"
            )
            raise FileNotFoundError()

        self.detector = StampDetector(detection_weight, device=device)
        self.padding = 3

    def __call__(self, image_list: Union[List[npt.NDArray], npt.NDArray], batch_size: int = 16) -> List[npt.NDArray]:
        """Detect and remove stamps from document images

        Args:
            image_list (Union[List[npt.NDArray], npt.NDArray]): list of input images
            batch_size (int, optional): Defaults to 16.

        Returns:
            List[np.ndarray]: Input images with stamps removed
        """
        if not isinstance(image_list, (np.ndarray, list)):
            raise TypeError("Invalid Type: Input must be of type list or np.ndarray")

        if len(image_list) > 0:
            check_image_shape(image_list[0])
        else:
            return []
        return self.__batch_removing(image_list, batch_size)  # type:ignore

    def __batch_removing(self, image_list, batch_size=16):  # type: ignore
        new_pages = []

        shapes = set(list(x.shape for x in image_list))
        images_batch, indices = create_batch(image_list, shapes, batch_size)
        # num_batch = len(image_list) // batch_size
        detection_predictions = []
        for batch in images_batch:
            if len(batch):
                detection_predictions.extend(self.detector(batch))
        z = zip(detection_predictions, indices)
        sorted_result = sorted(z, key=lambda x: x[1])
        detection_predictions, _ = zip(*sorted_result)
        for idx, page_boxes in enumerate(detection_predictions):
            page_img = image_list[idx]
            h, w, c = page_img.shape
            for box in page_boxes:
                x_min, y_min, x_max, y_max = box[:4]
                stamp_area = page_img[
                    max(y_min - self.padding, 0) : min(y_max + self.padding, h),
                    max(x_min - self.padding, 0) : min(x_max + self.padding, w),
                ]
                stamp_area = self.remover([stamp_area])  # type:ignore

                page_img[
                    max(y_min - self.padding, 0) : min(y_max + self.padding, h),
                    max(x_min - self.padding, 0) : min(x_max + self.padding, w),
                    :,
                ] = stamp_area[0]
            new_pages.append(page_img)
        return new_pages
