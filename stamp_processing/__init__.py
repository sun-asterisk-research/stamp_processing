"""
    isort: skip_file
"""
import sys

from stamp_processing.module.yolov5 import YOLO_DIR
from pkg_resources import DistributionNotFound, get_distribution

__version__ = None
try:
    __version__ = get_distribution("table_reconstruction").version
except DistributionNotFound:
    __version__ == "0.0.0"  # package is not installed
    pass

sys.path.append(str(YOLO_DIR))

from stamp_processing.detector import StampDetector
from stamp_processing.remover import StampRemover
