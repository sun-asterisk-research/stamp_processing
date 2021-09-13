import os
import shutil

from functools import partial
from typing import List, Union

import numpy as np
import torch

from stamp_processing.preprocess import create_batch, process_image
from stamp_processing.module.yolov5.utils.datasets import letterbox
from stamp_processing.module.yolov5.utils.general import non_max_suppression, scale_coords
from stamp_processing.utils import *


class StampDetector:
    def __init__(self, model_path=None, device="cpu", conf_thres=0.3, iou_thres=0.3):
        assert device == "cpu", "Currently only support cpu inference"

        try:
            if model_path is None:
                logger.info("Downloading stamp detection weight from google drive")
                download_weight(DETECTOR_WEIGHT_ID, output="stamp_detector.pt")

                if not os.path.exists("tmp/"):
                    os.makedirs("tmp/", exist_ok=True)

                model_path = os.path.join("/tmp/", "stamp_detector.pt")
                shutil.move("stamp_detector.pt", model_path)
                logger.info(f"Finished downloading. Weight is saved at {model_path}")

            self.device = select_device(device)
            self.model, self.stride = load_yolo_model(model_path, device=device)
        except Exception as e:
            logger.error(e)
            logger.error("There is something wrong when loading detector weight")
            logger.error(
                f"""Please make sure you provide the correct path to the weight
                or mannually download the weight at https://drive.google.com/file/d/{DETECTOR_WEIGHT_ID}/view?usp=sharing"""
            )
            raise ValueError()
        print("Using {} for stamp detection".format(self.device))

        self.img_size = 640
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.process_func_ = partial(process_image, device=self.device)

    def __call__(self, image_list: Union[List[np.ndarray], np.ndarray]) -> List[np.ndarray]:
        """
        Returns a list of bounding boxes [xmin, ymin, xmax, ymax] for each image in image_list
        Each element in the list is a numpy array of shape N x 4

        Args:
            image_list (List[np.array]): input images

        Returns:
            [List[np.ndarray]]: output bounding boxes
        """
        if not isinstance(image_list, (np.ndarray, list)):
            raise TypeError("Invalid Type: Input must be of type list or np.ndarray")

        if len(image_list) > 0:
            check_image_shape(image_list[0])
        else:
            return []
        return self.detect(image_list)

    def detect(self, image_list):

        batches, indices = create_batch(image_list, set(list(x.shape for x in image_list)))
        predictions = []

        for origin_images in batches:
            images = [letterbox(x, 640, stride=32)[0] for x in origin_images]
            images = list(map(self.process_func_, images))
            tensor = torch.stack(images)

            with torch.no_grad():
                pred = self.model(tensor)[0]
            all_boxes = []
            pred = non_max_suppression(pred, 0.3, 0.30, classes=0, agnostic=1)

            for idx, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(images[idx].shape[1:], det[:, :4], origin_images[0].shape).round()
                    det = det[:, :4]
                    all_boxes.append(det.cpu().numpy().astype("int").tolist())
                else:
                    all_boxes.append([])

            predictions.extend(all_boxes)

        z = zip(predictions, indices)
        sorted_result = sorted(z, key=lambda x: x[1])
        predictions, _ = zip(*sorted_result)

        return list(predictions)
