import os
import shutil
import numpy as np

from typing import List

from stamp_processing.module.unet import UnetInference
from stamp_processing.detector import StampDetector
from stamp_processing.preprocess import create_batch
from stamp_processing.utils import download_weight, REMOVER_WEIGHT_ID


class StampRemover:
    def __init__(self, detection_weight=None, removal_weight=None, device=""):

        if removal_weight is None:
            print("Downloading stamp remover weight from google drive")
            download_weight(REMOVER_WEIGHT_ID, output="stamp_remover.pkl")

            if not os.path.exists("tmp/"):
                os.makedirs("tmp/", exist_ok=True)

            removal_weight = os.path.join("/tmp/", "stamp_remover.pkl")
            shutil.move("stamp_remover.pkl", removal_weight)
            print(f"Finished downloading. Weight is saved at {removal_weight}")

        self.remover = UnetInference(removal_weight)
        self.detector = StampDetector(device=device)
        self.padding = 3

    def __call__(self, image_list: List[np.ndarray], batch_size=16) -> List[np.ndarray]:
        """
        Detect and remove stamps from document images
        Args:
            image_list (List[np.ndarray]): list of input images
            batch_size (int, optional): Defaults to 16.

        Returns:
            List[np.ndarray]: Input images with stamps removed
        """
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
