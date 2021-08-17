import numpy as np
import torch

from functools import partial
from typing import List

from preprocess import letterbox, process_image, create_batch
from postprocess import non_max_suppression, scale_coords
from utils import select_device, load_torch_script_model



class StampDetector:
    def __init__(self, model_path, device="cpu", conf_thres=0.3, iou_thres=0.3):
        self.device = select_device(device)
        # print("Using {} for stamp detection".format(self.device))
        self.model, self.stride = load_torch_script_model(model_path, device=self.device)

        self.img_size = 640
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.process_func_ = partial(process_image, device=self.device)

    def predict(self, image_list: List[np.array]):
        """
        Returns a list of bounding boxes [xmin, ymin, xmax, ymax] for each image in image_list

        Args:
            image_list (List[np.array]): input images

        Returns:
            [List[np.array]]: output bounding boxes
        """
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
                    det[:, :4] = scale_coords(
                        images[idx].shape[1:], det[:, :4], origin_images[0].shape
                    ).round()
                    det = det[:, :4]
                    all_boxes.append(det.cpu().numpy().astype("int").tolist())
                else:
                    all_boxes.append([])

            predictions.extend(all_boxes)

        z = zip(predictions, indices)
        sorted_result = sorted(z, key=lambda x: x[1])
        predictions, _ = zip(*sorted_result)

        return list(predictions)

