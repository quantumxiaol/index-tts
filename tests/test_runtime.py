import os
import unittest
from unittest.mock import patch

from indextts.utils.runtime import configure_mps_environment


_MPS_ENV_NAMES = (
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
)


class MPSRuntimeConfigurationTests(unittest.TestCase):
    def test_non_macos_is_a_noop(self):
        with (
            patch("indextts.utils.runtime.platform.system", return_value="Linux"),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = configure_mps_environment()
            self.assertIsNone(result)
            self.assertTrue(all(name not in os.environ for name in _MPS_ENV_NAMES))

    def test_macos_sets_conservative_defaults(self):
        with (
            patch("indextts.utils.runtime.platform.system", return_value="Darwin"),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = configure_mps_environment()
            self.assertEqual(result["PYTORCH_ENABLE_MPS_FALLBACK"], "1")
            self.assertEqual(result["PYTORCH_MPS_LOW_WATERMARK_RATIO"], "0.4")
            self.assertEqual(result["PYTORCH_MPS_HIGH_WATERMARK_RATIO"], "0.6")

    def test_user_values_take_precedence(self):
        existing = {
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.5",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.7",
        }
        with (
            patch("indextts.utils.runtime.platform.system", return_value="Darwin"),
            patch.dict(os.environ, existing, clear=True),
        ):
            result = configure_mps_environment()
            self.assertEqual(result, existing)


if __name__ == "__main__":
    unittest.main()
