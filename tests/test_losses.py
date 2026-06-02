import unittest


try:
    import mlx.core as mx

    from mlx_embeddings_lora.trainer.contrastive_trainer import (
        create_in_batch_negatives,
        gist_embed_loss,
        infonce_loss,
        multiple_negatives_ranking_loss,
        triplet_loss,
    )
except RuntimeError as exc:
    if "No Metal device available" not in str(exc):
        raise
    raise unittest.SkipTest("MLX Metal device is not available")


def assert_mx_allclose(testcase, actual, expected, atol=1e-6):
    mx.eval(actual, expected)
    testcase.assertTrue(bool(mx.all(mx.abs(actual - expected) < atol).item()))


class ContrastiveLossTest(unittest.TestCase):
    def test_infonce_matches_two_logit_cross_entropy(self):
        anchor = mx.array([[1.0, 0.0], [0.0, 1.0]])
        positive = mx.array([[1.0, 0.0], [0.0, 1.0]])
        negative = mx.array([[0.0, 1.0], [1.0, 0.0]])

        losses = infonce_loss(anchor, positive, negative, temperature=1.0)
        expected = -mx.array([1.0, 1.0]) + mx.logsumexp(
            mx.array([[1.0, 0.0], [1.0, 0.0]]),
            axis=-1,
        )

        assert_mx_allclose(self, losses, expected)

    def test_triplet_loss_uses_margin_on_cosine_distance(self):
        anchor = mx.array([[1.0, 0.0], [1.0, 0.0]])
        positive = mx.array([[1.0, 0.0], [0.0, 1.0]])
        negative = mx.array([[0.0, 1.0], [1.0, 0.0]])

        losses = triplet_loss(anchor, positive, negative, margin=0.5)

        assert_mx_allclose(self, losses, mx.array([0.0, 1.5]))

    def test_multiple_negatives_ranking_returns_per_sample_losses(self):
        anchor = mx.array([[1.0, 0.0], [0.0, 1.0]])
        positive = mx.array([[1.0, 0.0], [0.0, 1.0]])
        negative = mx.array([[0.0, 1.0], [1.0, 0.0]])

        losses = multiple_negatives_ranking_loss(anchor, positive, negative)

        self.assertEqual(losses.shape, (2,))
        expected = -mx.array([1.0, 1.0]) + mx.logsumexp(
            mx.array([[1.0, 0.0], [1.0, 0.0]]),
            axis=-1,
        )
        assert_mx_allclose(self, losses, expected)

    def test_create_in_batch_negatives_rotates_positive_embeddings(self):
        anchor = mx.array([[10.0], [20.0], [30.0]])
        positive = mx.array([[1.0], [2.0], [3.0]])

        negatives = create_in_batch_negatives(anchor, positive)

        assert_mx_allclose(self, negatives, mx.array([[2.0], [3.0], [1.0]]))

    def test_gist_uses_row_positive_score_for_guided_filtering(self):
        anchor = mx.array([[1.0, 0.0], [0.0, 1.0]])
        positive = mx.array([[1.0, 0.0], [0.0, 1.0]])

        # Row 0 keeps its off-diagonal negative. Row 1 filters its off-diagonal
        # because the guide scores it above that row's assigned positive.
        guide_anchor = mx.array([[1.0, 0.0], [0.8, 0.6]])
        guide_positive = mx.array([[1.0, 0.0], [0.0, 1.0]])

        losses = gist_embed_loss(
            anchor,
            positive,
            guide_anchor,
            guide_positive,
            temperature=1.0,
            guide_threshold=0.0,
        )

        expected_row_0 = -1.0 + mx.logsumexp(mx.array([1.0, 0.0]), axis=-1)
        assert_mx_allclose(self, losses, mx.array([expected_row_0, 0.0]))


if __name__ == "__main__":
    unittest.main()
