import torch 
import os

from torch import nn

from models.experimental import attempt_load
from models.common import Conv
from models.experimental import attempt_load
from models.yolo import Detect


from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


def save_exported_script(model, sample: torch.Tensor, path: str) -> None:
    traced_script_module = torch.jit.trace(model, sample)
    traced_script_module.save(path)


def convert_yolo_model(weight_path: str, export_dir: str, export_name: str) -> None:
    model = attempt_load(weight_path, map_location='cpu')
    img = torch.rand((1, 3, 640, 640), dtype=torch.float32)

    for _, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = nn.Hardswish()
            elif isinstance(m.act, nn.SiLU):

                m.act = nn.SiLU()
        elif isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = False

    for _ in range(2):
        model(img)

    save_exported_script(
        model, img, os.path.join(export_dir, "{}.pt".format(export_name)),
    )


if __name__ == '__main__':
    convert_yolo_model('weight/stamp_detect.pt', 'weight', 'traced_weight')

