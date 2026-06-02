import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from mlx_embeddings_lora.trainer.contrastive_trainer import gist_embed_loss


class GISTEmbedLossTest(unittest.TestCase):
    def test_uses_row_positive_score_for_guided_negative_filtering(self):
        anchor = mx.array([[1.0, 0.0], [0.0, 1.0]])
        positive = mx.array([[1.0, 0.0], [0.0, 1.0]])

        # Row 0: guide positive score is 1.0, off-diagonal score is 0.0, so the
        # off-diagonal remains a valid negative.
        # Row 1: guide positive score is 0.6, off-diagonal score is 0.8, so the
        # off-diagonal is filtered as a likely false negative.
        guide_anchor = mx.array([[1.0, 0.0], [0.6, 0.8]])
        guide_positive = mx.array([[1.0, 0.0], [0.0, 1.0]])

        losses = gist_embed_loss(
            anchor,
            positive,
            guide_anchor,
            guide_positive,
            temperature=1.0,
            guide_threshold=0.0,
        )
        mx.eval(losses)

        expected_row_0 = -1.0 + mx.logsumexp(mx.array([1.0, 0.0]), axis=-1)
        expected = mx.array([expected_row_0, 0.0])
        self.assertTrue(bool(mx.all(mx.abs(losses - expected) < 1e-6).item()))


if __name__ == "__main__":
    unittest.main()
