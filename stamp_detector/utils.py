import os
import torch


def select_device(device=""):
    cpu = device.lower() == "cpu"
    cuda = not cpu and torch.cuda.is_available()
    return torch.device("cuda:0" if cuda else "cpu")


def load_torch_script_model(weight_path, device="cpu"):
    model = torch.jit.load(weight_path, map_location=device)
    stride = 32
    return model, stride