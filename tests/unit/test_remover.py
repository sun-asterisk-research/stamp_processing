import os
import unittest
import numpy as np
from stamp_processing.remover import StampRemover

class RemoverTestCase(unittest.TestCase):
    def test_ignore_gpu(self):
        with self.assertRaisesRegex(
            Exception,
            "Currently only support cpu inference",
        ):
            StampRemover(None, None, "gpu")

    def test_output_size(self):
        image = np.ones((224, 224, 3))
        
        remover = StampRemover()
        out = remover([image])[0]
        
        self.assertEqual(image.shape, out.shape)

if __name__ == "__main__":
    unittest.main()
