import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from stamp_processing.utils import (
    DETECTOR_WEIGHT_ID,
    check_image_shape,
    download_weight,
    select_device,
)


class UtilsTestCase(unittest.TestCase):
    def test_download_weight(self):
        try:
            download_weight(DETECTOR_WEIGHT_ID, "/tmp/test_utils/test.pt")
        except Exception as e:
            self.fail(e)

        self.assertFalse(os.path.exists("/tmp/test_utils/test_utilstest.pt"), "Missing downloaded file")

    def test_select_device(self):
        self.assertEqual(select_device("cpu"), torch.device("cpu"))

    def test_check_valid_image_shape(self):
        try:
            valid_image = np.random.randint(255, size=(900, 800, 3), dtype=np.uint8)
            check_image_shape(valid_image)
        except Exception as e:
            self.fail(e)

    def test_check_invalid_image_type(self):
        self.assertRaises(TypeError, check_image_shape, None)

    def test_check_invalid_image_shape(self):
        invalid_image = np.random.randint(255, size=(900, 800), dtype=np.uint8)
        self.assertRaises(ValueError, check_image_shape, invalid_image)

    def test_check_image_has_wrong_dimensional_number(self):
        invalid_image = np.random.randint(255, size=(900, 800, 4), dtype=np.uint8)
        self.assertRaises(ValueError, check_image_shape, invalid_image)

if __name__ == "__main__":
    unittest.main()