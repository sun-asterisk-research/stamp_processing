import unittest

import torch

from stamp_detector import utils


class UtilsTestCase(unittest.TestCase):
    def test_select_device(self):
        self.assertEqual(utils.select_device("cpu"), torch.device("cpu"))
