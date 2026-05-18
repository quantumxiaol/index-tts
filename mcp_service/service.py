import os
import platform
import shutil
import threading
import uuid
from typing import Any, Dict, Optional, Sequence

if platform.system() == "Darwin":
    # Configure MPS to fall back to CPU and avoid memory pressure on macOS.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.6"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
    print(">> [System] macOS detected; applied MPS memory watermarks (Low=0.6, High=0.8).")

import httpx
from mcp.server import FastMCP

from indextts.infer_v2_5 import IndexTTS2


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


MODEL_DIR = os.getenv("INDEXTTS_MODEL_DIR", "checkpoints")
CONFIG_PATH = os.getenv("INDEXTTS_CONFIG_PATH", "checkpoints/config.yaml")
DEVICE = os.getenv("INDEXTTS_DEVICE") or None

USE_BF16 = _env_flag("INDEXTTS_USE_BF16", default=False)
USE_CUDA_KERNEL = _env_flag("INDEXTTS_USE_CUDA_KERNEL", default=False)
USE_DEEPSPEED = _env_flag("INDEXTTS_USE_DEEPSPEED", default=False)
USE_QWEN_EMO = _env_flag("INDEXTTS_USE_QWEN_EMO", default=True)
USE_ACCEL = _env_flag("INDEXTTS_USE_ACCEL", default=False)
USE_TORCH_COMPILE = _env_flag("INDEXTTS_USE_TORCH_COMPILE", default=False)

AUDIO_IN_DIR = os.getenv("TTS_INPUT_DIR", "inputs")
AUDIO_OUT_DIR = os.getenv("TTS_OUTPUT_DIR", "outputs")
SUPPORTED_LANGUAGES = {"ZH", "EN", "JA", "ES", "AR"}


mcp = FastMCP(name="IndexTTS2.5MCP")
tts = IndexTTS2(
    cfg_path=CONFIG_PATH,
    model_dir=MODEL_DIR,
    use_bf16=USE_BF16,
    use_cuda_kernel=USE_CUDA_KERNEL,
    use_deepspeed=USE_DEEPSPEED,
    use_qwen_emo=USE_QWEN_EMO,
    use_accel=USE_ACCEL,
    use_torch_compile=USE_TORCH_COMPILE,
    device=DEVICE,
)
_tts_lock = threading.Lock()


def _ensure_dirs() -> None:
    os.makedirs(AUDIO_IN_DIR, exist_ok=True)
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)


def _copy_to_audio_in(path_value: str) -> str:
    _ensure_dirs()
    suffix = os.path.splitext(path_value)[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{suffix}"
    dst_path = os.path.abspath(os.path.join(AUDIO_IN_DIR, filename))
    shutil.copy2(path_value, dst_path)
    return dst_path


def _download_to_audio_in(url: str) -> str:
    _ensure_dirs()
    suffix = os.path.splitext(url.split("?")[0])[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{suffix}"
    dst_path = os.path.abspath(os.path.join(AUDIO_IN_DIR, filename))
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        with open(dst_path, "wb") as f:
            f.write(resp.content)
    return dst_path


def _resolve_audio_input(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return _download_to_audio_in(path_or_url)
    abs_path = os.path.abspath(path_or_url)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"prompt audio not found: {abs_path}")
    abs_inputs = os.path.abspath(AUDIO_IN_DIR)
    if abs_path.startswith(abs_inputs + os.sep):
        return abs_path
    return _copy_to_audio_in(abs_path)


def _resolve_optional_audio(path_or_url: Optional[str]) -> Optional[str]:
    if not path_or_url:
        return None
    return _resolve_audio_input(path_or_url)


def _resolve_text_file(path_value: str) -> str:
    abs_path = os.path.abspath(path_value)
    if os.path.exists(abs_path):
        return abs_path
    candidate = os.path.abspath(os.path.join(AUDIO_IN_DIR, path_value))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"text file not found: {abs_path}")


def _normalize_output_name(output_name: Optional[str]) -> str:
    if output_name:
        name = os.path.basename(output_name)
        if name:
            _, ext = os.path.splitext(name)
            if not ext:
                return f"{name}.wav"
            return name
    return f"{uuid.uuid4().hex}.wav"


def _build_output_path(output_name: Optional[str]) -> str:
    _ensure_dirs()
    filename = _normalize_output_name(output_name)
    return os.path.abspath(os.path.join(AUDIO_OUT_DIR, filename))


def _normalize_language(language: str) -> str:
    normalized = language.strip().upper()
    if normalized not in SUPPORTED_LANGUAGES:
        choices = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"lang must be one of: {choices}.")
    return normalized


def _validate_options(
    *,
    emo_alpha: float,
    emo_vector: Optional[Sequence[float]],
    interval_silence: int,
    max_text_tokens_per_segment: int,
    duration_factor: float,
) -> None:
    if not 0.0 <= emo_alpha <= 1.0:
        raise ValueError("emo_alpha must be between 0.0 and 1.0.")
    if emo_vector is not None and len(emo_vector) != 8:
        raise ValueError("emo_vector must contain 8 float values.")
    if interval_silence < 0:
        raise ValueError("interval_silence must be non-negative.")
    if max_text_tokens_per_segment < 1:
        raise ValueError("max_text_tokens_per_segment must be positive.")
    if not 0.5 <= duration_factor <= 2.0:
        raise ValueError("duration_factor must be between 0.5 and 2.0.")


def _get_sample_rate() -> Optional[int]:
    try:
        return int(tts.cfg.s2mel["preprocess_params"]["sr"])
    except (KeyError, TypeError, ValueError):
        return None


@mcp.tool(
    name="tts_synthesize",
    description="Synthesize speech with IndexTTS2.5 using a local path or URL prompt audio.",
)
async def tts_synthesize(
    text: str,
    prompt_wav_path: str,
    lang: str = "ZH",
    output_name: Optional[str] = None,
    emo_audio_prompt: Optional[str] = None,
    emo_alpha: float = 1.0,
    emo_vector: Optional[Sequence[float]] = None,
    use_emo_text: bool = False,
    emo_text: Optional[str] = None,
    use_random: bool = False,
    interval_silence: int = 200,
    max_text_tokens_per_segment: int = 120,
    duration_factor: float = 1.0,
    text_normalization: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    _validate_options(
        emo_alpha=emo_alpha,
        emo_vector=emo_vector,
        interval_silence=interval_silence,
        max_text_tokens_per_segment=max_text_tokens_per_segment,
        duration_factor=duration_factor,
    )
    language = _normalize_language(lang)
    prompt_path = _resolve_audio_input(prompt_wav_path)
    emo_prompt_path = _resolve_optional_audio(emo_audio_prompt)
    out_path = _build_output_path(output_name)
    effective_use_emo_text = use_emo_text or (emo_text is not None)
    with _tts_lock:
        tts.infer(
            spk_audio_prompt=prompt_path,
            text=text,
            output_path=out_path,
            lang=language,
            emo_audio_prompt=emo_prompt_path,
            emo_alpha=emo_alpha,
            emo_vector=list(emo_vector) if emo_vector is not None else None,
            use_emo_text=effective_use_emo_text,
            emo_text=emo_text,
            use_random=use_random,
            interval_silence=interval_silence,
            verbose=verbose,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            duration_factor=duration_factor,
            text_normalization=text_normalization,
        )
    return {
        "status": "success",
        "audio_path": out_path,
        "prompt_audio_path": os.path.abspath(prompt_path),
        "emo_audio_path": os.path.abspath(emo_prompt_path) if emo_prompt_path else None,
        "sample_rate": _get_sample_rate(),
    }


@mcp.tool(
    name="tts_batch_file",
    description="Batch synthesize each non-empty line of a text file with IndexTTS2.5.",
)
async def tts_batch_file(
    text_file: str,
    prompt_wav_path: str,
    lang: str = "ZH",
    output_prefix: Optional[str] = None,
    emo_audio_prompt: Optional[str] = None,
    emo_alpha: float = 1.0,
    emo_vector: Optional[Sequence[float]] = None,
    use_emo_text: bool = False,
    emo_text: Optional[str] = None,
    use_random: bool = False,
    interval_silence: int = 200,
    max_text_tokens_per_segment: int = 120,
    duration_factor: float = 1.0,
    text_normalization: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    _validate_options(
        emo_alpha=emo_alpha,
        emo_vector=emo_vector,
        interval_silence=interval_silence,
        max_text_tokens_per_segment=max_text_tokens_per_segment,
        duration_factor=duration_factor,
    )
    language = _normalize_language(lang)
    text_path = _resolve_text_file(text_file)
    prompt_path = _resolve_audio_input(prompt_wav_path)
    emo_prompt_path = _resolve_optional_audio(emo_audio_prompt)
    if output_prefix:
        output_base = os.path.basename(output_prefix)
        base_name = os.path.splitext(output_base)[0] or output_base
    else:
        base_name = os.path.splitext(os.path.basename(text_path))[0]
    output_paths = []
    effective_use_emo_text = use_emo_text or (emo_text is not None)

    with _tts_lock:
        with open(text_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                output_name = f"{base_name}_{line_no}.wav"
                out_path = _build_output_path(output_name)
                tts.infer(
                    spk_audio_prompt=prompt_path,
                    text=text,
                    output_path=out_path,
                    lang=language,
                    emo_audio_prompt=emo_prompt_path,
                    emo_alpha=emo_alpha,
                    emo_vector=list(emo_vector) if emo_vector is not None else None,
                    use_emo_text=effective_use_emo_text,
                    emo_text=emo_text,
                    use_random=use_random,
                    interval_silence=interval_silence,
                    verbose=verbose,
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    duration_factor=duration_factor,
                    text_normalization=text_normalization,
                )
                output_paths.append(out_path)
    return {
        "status": "success",
        "audio_paths": output_paths,
        "prompt_audio_path": os.path.abspath(prompt_path),
        "emo_audio_path": os.path.abspath(emo_prompt_path) if emo_prompt_path else None,
        "text_file": text_path,
        "sample_rate": _get_sample_rate(),
    }
