import unittest

import mlx_embeddings_lora


class PackageMetadataTest(unittest.TestCase):
    def test_version_is_exported(self):
        self.assertIsInstance(mlx_embeddings_lora.__version__, str)
        self.assertNotEqual(mlx_embeddings_lora.__version__, "")


if __name__ == "__main__":
    unittest.main()
