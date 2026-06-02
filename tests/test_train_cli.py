import unittest


try:
    from mlx_embeddings_lora import train
except RuntimeError as exc:
    if "No Metal device available" not in str(exc):
        raise
    raise unittest.SkipTest("MLX Metal device is not available")


class TrainCliTest(unittest.TestCase):
    def test_parser_supports_gist_mode_and_guide_options(self):
        args = train.build_parser().parse_args(
            [
                "--train-mode",
                "gist",
                "--guide-model",
                "guide-model",
                "--guide-threshold",
                "0.05",
            ]
        )

        self.assertEqual(args.train_mode, "gist")
        self.assertEqual(args.guide_model, "guide-model")
        self.assertEqual(args.guide_threshold, 0.05)

    def test_config_defaults_keep_gist_margin_at_zero(self):
        self.assertEqual(train.CONFIG_DEFAULTS["guide_threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
