import sys

from stamp_processing.module.yolov5 import YOLO_DIR


sys.path.append(str(YOLO_DIR))

from stamp_processing.detector import StampDetector
from stamp_processing.remover import StampRemover
