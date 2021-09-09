import cv2
import numpy as np
import torch


def create_batch(images: np.array, shapes: set, batch_size: int = 16):
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
        images_mini_batch = []
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


def process_image(img, device="cpu"):
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


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
