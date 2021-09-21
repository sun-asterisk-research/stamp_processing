import unittest

import numpy as np
import torch
from stamp_processing.preprocess import create_batch, process_image


class PreprocessTestCase(unittest.TestCase):
    def test_create_batch(self):
        images = [
            np.random.randint(255, size=(640, 530, 4), dtype=np.uint8),
            np.random.randint(255, size=(640, 300, 4), dtype=np.uint8),
            np.random.randint(255, size=(640, 530, 4), dtype=np.uint8),
            np.random.randint(255, size=(640, 530, 4), dtype=np.uint8),
            np.random.randint(255, size=(640, 300, 4), dtype=np.uint8),
            np.random.randint(255, size=(640, 530, 4), dtype=np.uint8),
        ]

        images_batch, indices = create_batch(images, set(list(x.shape for x in images)), batch_size=2)
        self.assertEqual(sum(len(batch) for batch in images_batch), len(images))
        self.assertListEqual([1, 4, 0, 2, 3, 5], indices)

    def test_process_image(self):
        image = np.random.randint(255, size=(640, 530, 3), dtype=np.uint8)
        output = process_image(image)
        self.assertFalse(output.is_cuda)
        self.assertListEqual(list(output.shape), [3, 640, 640])
        self.assertTrue(output.dtype == torch.float32)

if __name__ == "__main__":
    unittest.main()