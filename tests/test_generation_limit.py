import re
import types
import unittest
from unittest.mock import Mock

import torch

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.exceptions import GenerationLengthExceededError


class _TokenizerStub:
    def encode(self, text, **kwargs):
        return [2, 3]


class _GPTStub:
    def __init__(self, stop_mel_token):
        self.stop_mel_token = stop_mel_token
        self.do_sample = None

    def merge_emovec(self, *args, **kwargs):
        return torch.zeros(1, 8)

    def inference_speech(self, *args, **kwargs):
        self.do_sample = kwargs["do_sample"]
        codes = torch.tensor([[10, 11, 12]])
        latent = torch.zeros(1, 4, 8)
        return codes, latent


def _strict_limit_stub():
    tts = IndexTTS2.__new__(IndexTTS2)
    tts.device = "cpu"
    tts.dtype = None
    tts.stop_mel_token = 999
    tts.gr_progress = None

    prompt = "prompt.wav"
    tts.cache_spk_audio_prompt = prompt
    tts.cache_spk_cond = torch.zeros(1, 4, 8)
    tts.cache_s2mel_style = torch.zeros(1, 192)
    tts.cache_s2mel_prompt = torch.zeros(1, 4, 8)
    tts.cache_mel = torch.zeros(1, 80, 4)
    tts.cache_emo_audio_prompt = prompt
    tts.cache_emo_cond = torch.zeros(1, 4, 8)

    tts.text_process = types.SimpleNamespace(
        clean_pattern=re.compile(r"(?!x)x"),
        char_rep_map={},
    )
    tts.tokenizer = _TokenizerStub()
    tts.ja_text_process = types.SimpleNamespace(process_ja_text=lambda text: text)
    tts.split_text_by_tokens = lambda text, max_tokens, lang_prefix: [text]
    tts.gpt = _GPTStub(tts.stop_mel_token)
    tts.semantic_codec = types.SimpleNamespace(decode=Mock(side_effect=AssertionError("must not decode")))
    return tts, prompt


class GenerationLimitTests(unittest.TestCase):
    def test_strict_limit_stops_before_s2mel_and_forwards_sampling_mode(self):
        tts, prompt = _strict_limit_stub()
        generator = tts.infer_generator(
            spk_audio_prompt=prompt,
            text="测试",
            output_path=None,
            lang="ZH",
            text_normalization=False,
            max_mel_tokens=3,
            raise_on_max_mel_tokens=True,
            do_sample=False,
        )

        with self.assertRaises(GenerationLengthExceededError):
            next(generator)

        self.assertFalse(tts.gpt.do_sample)
        tts.semantic_codec.decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
