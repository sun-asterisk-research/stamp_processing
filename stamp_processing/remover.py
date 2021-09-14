import os
import shutil
import numpy as np

from typing import List, Union

from stamp_processing.module.unet import UnetInference
from stamp_processing.detector import StampDetector
from stamp_processing.preprocess import create_batch
from stamp_processing.utils import *


class StampRemover:
    def __init__(self, detection_weight=None, removal_weight=None, device="cpu"):
        assert device == "cpu", "Currently only support cpu inference"

        if removal_weight is None:
            if not os.path.exists("tmp/"):
                os.makedirs("tmp/", exist_ok=True)
            removal_weight = os.path.join("tmp", "stamp_remover.pkl")

            logger.info("Downloading stamp remover weight from google drive")
            download_weight(REMOVER_WEIGHT_ID, output=removal_weight)
            logger.info(f"Finished downloading. Weight is saved at {removal_weight}")

        try:
            self.remover = UnetInference(removal_weight)
        except Exception as e:
            logger.error(e)
            logger.error("There is something wrong when loading detector weight")
            logger.error(
                f"""Please make sure you provide the correct path to the weight
                or mannually download the weight at https://drive.google.com/file/d/{REMOVER_WEIGHT_ID}/view?usp=sharing"""
            )
            raise ValueError()

        self.detector = StampDetector(detection_weight, device=device)
        self.padding = 3

    def __call__(self, image_list: Union[List[np.ndarray], np.ndarray], batch_size=16) -> List[np.ndarray]:
        """
        Detect and remove stamps from document images
        Args:
            image_list (List[np.ndarray]): list of input images
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
        return self.__batch_removing(image_list, batch_size)

    def __batch_removing(self, image_list, batch_size=16):
        new_pages = []

        shapes = set(list(x.shape for x in image_list))
        images_batch, indices = create_batch(image_list, shapes, batch_size)
        num_batch = len(image_list) // batch_size
        detection_predictions = []
        for batch in images_batch:
            if len(batch):
                detection_predictions.extend(self.detector.detect(batch))
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

                stamp_area = self.remover([stamp_area])
                page_img[
                    max(y_min - self.padding, 0) : min(y_max + self.padding, h),
                    max(x_min - self.padding, 0) : min(x_max + self.padding, w),
                    :,
                ] = stamp_area[0]
            new_pages.append(page_img)
        return new_pages
