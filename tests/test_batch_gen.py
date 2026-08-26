import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import batch_gen
from indextts.utils.exceptions import GenerationLengthExceededError


class FakeIndexTTS2:
    device = "cpu"
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []
        self.instances.append(self)

    def infer(self, *, output_path, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise GenerationLengthExceededError(
                max_mel_tokens=kwargs["max_mel_tokens"],
                input_text_tokens=10,
                max_text_tokens_per_segment=kwargs["max_text_tokens_per_segment"],
            )
        Path(output_path).write_bytes(b"fake-wav")
        return output_path


class BatchGenerationTests(unittest.TestCase):
    def test_length_limit_retries_with_sampling(self):
        FakeIndexTTS2.instances.clear()
        with tempfile.TemporaryDirectory(prefix="indextts-batch-test-") as tmp:
            root = Path(tmp)
            text_file = root / "input.txt"
            voice_file = root / "voice.wav"
            output_dir = root / "output"
            text_file.write_text("测试。\n", encoding="utf-8")
            voice_file.write_bytes(b"fake-voice")
            argv = [
                "batch_gen.py",
                str(text_file),
                "-v",
                str(voice_file),
                "-o",
                str(output_dir),
                "--device",
                "cpu",
                "--memory_diagnostics",
            ]

            with (
                patch.object(batch_gen, "IndexTTS2", FakeIndexTTS2),
                patch.object(batch_gen, "_release_device_cache") as release_cache,
                patch.object(sys, "argv", argv),
            ):
                batch_gen.main()

            self.assertEqual((output_dir / "input_1.wav").read_bytes(), b"fake-wav")
            # CPU does not clear on successful lines; the failed attempt still
            # clears so retries do not retain its generation tensors.
            self.assertEqual(release_cache.call_count, 1)
            calls = FakeIndexTTS2.instances[0].calls
            self.assertTrue(FakeIndexTTS2.instances[0].init_kwargs["memory_diagnostics"])
            self.assertEqual([call["do_sample"] for call in calls], [False, True])
            self.assertTrue(all(call["raise_on_max_mel_tokens"] for call in calls))


if __name__ == "__main__":
    unittest.main()
