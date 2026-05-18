import asyncio
import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class FakeIndexTTS2:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []
        self.cfg = types.SimpleNamespace(
            s2mel={"preprocess_params": {"sr": 22050}},
        )

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["output_path"]


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)

    def _import_service(self, module_name):
        fake_infer = types.ModuleType("indextts.infer_v2_5")
        fake_infer.IndexTTS2 = FakeIndexTTS2
        module_patch = patch.dict(
            sys.modules,
            {"indextts.infer_v2_5": fake_infer},
        )
        environment_patch = patch.dict(
            os.environ,
            {
                "TTS_INPUT_DIR": str(self.temp_path / "inputs"),
                "TTS_OUTPUT_DIR": str(self.temp_path / "outputs"),
            },
        )
        module_patch.start()
        environment_patch.start()
        self.addCleanup(module_patch.stop)
        self.addCleanup(environment_patch.stop)
        sys.modules.pop(module_name, None)
        self.addCleanup(sys.modules.pop, module_name, None)
        return importlib.import_module(module_name)

    def test_fastapi_synthesize_forwards_v25_options(self):
        service = self._import_service("fastapi_service.service")
        prompt_path = self.temp_path / "prompt.wav"
        prompt_path.touch()

        response = TestClient(service.app).post(
            "/tts/synthesize",
            json={
                "text": "Hello IndexTTS 2.5",
                "prompt_wav_path": str(prompt_path),
                "lang": "en",
                "duration_factor": 1.2,
                "text_normalization": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sample_rate"], 22050)
        self.assertIs(service.tts.init_kwargs["use_bf16"], False)
        self.assertIs(service.tts.init_kwargs["use_qwen_emo"], True)
        self.assertEqual(service.tts.calls[-1]["lang"], "EN")
        self.assertEqual(service.tts.calls[-1]["duration_factor"], 1.2)
        self.assertIs(service.tts.calls[-1]["text_normalization"], False)

    def test_fastapi_rejects_unsupported_language(self):
        service = self._import_service("fastapi_service.service")
        prompt_path = self.temp_path / "prompt.wav"
        prompt_path.touch()

        response = TestClient(service.app).post(
            "/tts/synthesize",
            json={
                "text": "hello",
                "prompt_wav_path": str(prompt_path),
                "lang": "DE",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(service.tts.calls, [])

    def test_mcp_synthesize_forwards_v25_options(self):
        service = self._import_service("mcp_service.service")
        prompt_path = self.temp_path / "prompt.wav"
        prompt_path.touch()

        result = asyncio.run(
            service.tts_synthesize(
                text="Hola IndexTTS 2.5",
                prompt_wav_path=str(prompt_path),
                lang="es",
                duration_factor=0.9,
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["sample_rate"], 22050)
        self.assertEqual(service.tts.calls[-1]["lang"], "ES")
        self.assertEqual(service.tts.calls[-1]["duration_factor"], 0.9)

    def test_fastapi_client_payload_includes_v25_options(self):
        from fastapi_service.client import _build_parser, _build_payload

        args = _build_parser().parse_args(
            [
                "synthesize",
                "--prompt-wav-path",
                "prompt.wav",
                "--text",
                "hello",
                "--lang",
                "EN",
                "--duration-factor",
                "1.1",
                "--no-text-normalization",
            ]
        )

        payload = _build_payload(args)
        self.assertEqual(payload["lang"], "EN")
        self.assertEqual(payload["duration_factor"], 1.1)
        self.assertIs(payload["text_normalization"], False)


if __name__ == "__main__":
    unittest.main()
