import torch
import gdown

from stamp_processing.module.yolov5.models.experimental import attempt_load
from stamp_processing.module.yolov5 import YOLO_DIR


DETECTOR_WEIGHT_URL = "https://drive.google.com/uc?id=1M3xY_DkDiA5Eg6Crgko5-rbyAEvo1X4D"
REMOVER_WEIGHT_URL = "https://drive.google.com/uc?id=1fQGVnatgHcMTmOxswqhE-vqoF_beovs1"


def select_device(device=""):
    cpu = device.lower() == "cpu"
    cuda = not cpu and torch.cuda.is_available()
    return torch.device("cuda:0" if cuda else "cpu")


def load_torch_script_model(weight_path, device="cpu"):
    model = torch.jit.load(weight_path, map_location=device)
    stride = 32
    return model, stride


def load_yolo_model(weight_path, device):
    print(YOLO_DIR)
    model = torch.hub.load(str(YOLO_DIR), "custom", path=weight_path, source="local")
    model.to(device)
    return model, model.stride


def download_weight(url, output=None, quiet=False):
    return gdown.download(url=url, output=output, quiet=quiet)
