import os
import shutil
import unittest
from unittest import mock
from stamp_processing.detector import StampDetector
import stamp_processing
from stamp_processing import utils
from unittest.mock import patch, MagicMock
import shutil
import numpy as np
import torch

torch.cuda.is_available = lambda: False


class DetectorTestCase(unittest.TestCase):
    def test_ignore_gpu(self):
        with self.assertRaisesRegex(
            Exception,
            "Currently only support cpu inference",
        ):
            StampDetector(None, "gpu")

    @patch.multiple(
        "stamp_processing.detector",
        download_weight=MagicMock(return_value=None),
        scale_coords=MagicMock(return_value=torch.FloatTensor([1, 2, 3, 4])),
        load_yolo_model=MagicMock(return_value=(torch.ones((1, 1, 1, 6), device=torch.device("cpu")), 32)),
        os=MagicMock(),
    )
    def test_detector(self, *mocks):
        detector = StampDetector(None, "cpu")
        self.assertEqual(detector.device.type, "cpu")
        with self.assertRaisesRegex(TypeError, "Invalid Type: Input must be of type list or np.ndarray"):
            detector(None)

        self.assertFalse(detector([]))

        image = np.random.randint(255, size=(900, 800, 3), dtype=np.uint8)
        detector.model = lambda _: torch.ones((1, 1, 1, 6), device=detector.device)
        detector([image])
        detector.model = lambda _: [torch.full((1, 1, 1, 6), 0, device=detector.device)]
        detector([image])

    @patch("stamp_processing.detector.download_weight", return_value=None)
    @patch("stamp_processing.detector.load_yolo_model", side_effect=Exception())
    @patch("stamp_processing.detector.os")
    def test_load_model_exception(self, download_weight_mock, load_yolo_model_mock, os_mock):
        with self.assertRaisesRegex(
            FileNotFoundError,
            "",
        ):
            os.makedirs.return_value = None
            StampDetector(None, "cpu")
            download_weight_mock.assert_called_once()
            load_yolo_model_mock.assert_called_once()

if __name__ == "__main__":
    unittest.main()