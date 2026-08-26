import unittest
from unittest.mock import patch

import torch

from indextts.utils.device import (
    DeviceMemorySnapshot,
    clear_device_cache,
    device_type,
    format_device_memory,
    get_device_memory_snapshot,
)
from indextts.utils.exceptions import GenerationLengthExceededError


class DeviceCacheTests(unittest.TestCase):
    def test_device_type_accepts_strings_and_torch_devices(self):
        self.assertEqual(device_type("cuda:1"), "cuda")
        self.assertEqual(device_type("mps"), "mps")
        self.assertEqual(device_type(torch.device("cpu")), "cpu")
        self.assertEqual(device_type(None), "cpu")

    @patch("indextts.utils.device.gc.collect")
    def test_cpu_collects_python_garbage_without_touching_accelerators(self, collect):
        cleared = clear_device_cache("cpu", collect_garbage=True, synchronize=True)

        self.assertFalse(cleared)
        collect.assert_called_once_with()

    def test_cuda_uses_cuda_backend(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            cleared = clear_device_cache("cuda:1", synchronize=True)

        self.assertTrue(cleared)
        synchronize.assert_called_once_with("cuda:1")
        empty_cache.assert_called_once_with()

    def test_mps_uses_argumentless_synchronize(self):
        with (
            patch.object(torch.mps, "is_available", return_value=True),
            patch.object(torch.mps, "synchronize") as synchronize,
            patch.object(torch.mps, "empty_cache") as empty_cache,
        ):
            cleared = clear_device_cache(torch.device("mps"), synchronize=True)

        self.assertTrue(cleared)
        synchronize.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_unavailable_backend_is_a_noop(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            cleared = clear_device_cache("cuda:0")

        self.assertFalse(cleared)
        empty_cache.assert_not_called()

    @unittest.skipUnless(hasattr(torch, "xpu"), "PyTorch build has no XPU module")
    def test_xpu_uses_xpu_backend(self):
        with (
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(torch.xpu, "synchronize") as synchronize,
            patch.object(torch.xpu, "empty_cache") as empty_cache,
        ):
            cleared = clear_device_cache("xpu:0", synchronize=True)

        self.assertTrue(cleared)
        synchronize.assert_called_once_with("xpu:0")
        empty_cache.assert_called_once_with()


class DeviceMemoryTests(unittest.TestCase):
    @patch("indextts.utils.device._process_rss_bytes", return_value=8 * 1024**3)
    def test_mps_snapshot_and_format_include_comparable_metrics(self, _process_rss):
        with (
            patch.object(torch.mps, "is_available", return_value=True),
            patch.object(torch.mps, "current_allocated_memory", return_value=6 * 1024**3),
            patch.object(torch.mps, "driver_allocated_memory", return_value=7 * 1024**3),
            patch.object(torch.mps, "recommended_max_memory", return_value=32 * 1024**3),
        ):
            snapshot = get_device_memory_snapshot("mps")

        message = format_device_memory(snapshot, "after GPT")
        self.assertEqual(snapshot.non_tensor_driver, 1024**3)
        self.assertIn("process RSS=8.00 GiB", message)
        self.assertIn("MPS tensors=6.00 GiB", message)
        self.assertIn("MPS driver=7.00 GiB", message)
        self.assertIn("MPS non-tensor driver=1.00 GiB", message)
        self.assertIn("MPS driver/recommended=21.9%", message)

    def test_cpu_format_does_not_report_accelerator_fields(self):
        snapshot = DeviceMemorySnapshot(device_type="cpu", process_rss=2 * 1024**3)
        message = format_device_memory(snapshot, "after model load")

        self.assertEqual(
            message,
            ">> [Memory] after model load: process RSS=2.00 GiB",
        )


class GenerationLengthExceededErrorTests(unittest.TestCase):
    def test_error_exposes_limit_context(self):
        error = GenerationLengthExceededError(
            max_mel_tokens=500,
            input_text_tokens=36,
            max_text_tokens_per_segment=50,
        )

        self.assertEqual(error.max_mel_tokens, 500)
        self.assertEqual(error.input_text_tokens, 36)
        self.assertIn("max_mel_tokens=500", str(error))


if __name__ == "__main__":
    unittest.main()
