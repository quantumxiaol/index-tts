import os
import shutil
import threading
import uuid
from typing import List, Optional, Sequence

from indextts.utils.runtime import configure_mps_environment

_mps_environment = configure_mps_environment()
if _mps_environment:
    print(
        ">> [System] macOS detected; MPS watermarks: "
        f"Low={_mps_environment['PYTORCH_MPS_LOW_WATERMARK_RATIO']}, "
        f"High={_mps_environment['PYTORCH_MPS_HIGH_WATERMARK_RATIO']}"
    )

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.device import clear_device_cache, device_type, log_device_memory
from indextts.utils.exceptions import GenerationLengthExceededError


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


app = FastAPI(title="IndexTTS2.5 FastAPI Service")
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
CLEAR_DEVICE_CACHE = _env_flag(
    "INDEXTTS_CLEAR_DEVICE_CACHE",
    default=device_type(tts.device) == "mps",
)
DEFAULT_MAX_MEL_TOKENS = int(os.getenv("INDEXTTS_MAX_MEL_TOKENS", "1500"))
if DEFAULT_MAX_MEL_TOKENS < 1:
    raise ValueError("INDEXTTS_MAX_MEL_TOKENS must be positive.")


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


def _get_sample_rate() -> Optional[int]:
    try:
        return int(tts.cfg.s2mel["preprocess_params"]["sr"])
    except (KeyError, TypeError, ValueError):
        return None


def _run_inference(**kwargs):
    try:
        return tts.infer(**kwargs)
    finally:
        if CLEAR_DEVICE_CACHE:
            clear_device_cache(tts.device, collect_garbage=True)
            log_device_memory(tts.device, "after cleanup")


class SynthesizeRequest(BaseModel):
    text: str
    prompt_wav_path: str
    lang: str = "ZH"
    output_name: Optional[str] = None
    emo_audio_prompt: Optional[str] = None
    emo_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    emo_vector: Optional[Sequence[float]] = None
    use_emo_text: bool = False
    emo_text: Optional[str] = None
    use_random: bool = False
    do_sample: bool = True
    interval_silence: int = Field(default=200, ge=0)
    max_text_tokens_per_segment: int = Field(default=120, ge=1)
    max_mel_tokens: int = Field(default=DEFAULT_MAX_MEL_TOKENS, ge=1)
    duration_factor: float = Field(default=1.0, ge=0.5, le=2.0)
    text_normalization: bool = True
    verbose: bool = False


class SynthesizeResponse(BaseModel):
    status: str
    audio_path: str
    prompt_audio_path: str
    emo_audio_path: Optional[str]
    sample_rate: Optional[int]


class BatchFileRequest(BaseModel):
    text_file: str
    prompt_wav_path: str
    lang: str = "ZH"
    output_prefix: Optional[str] = None
    emo_audio_prompt: Optional[str] = None
    emo_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    emo_vector: Optional[Sequence[float]] = None
    use_emo_text: bool = False
    emo_text: Optional[str] = None
    use_random: bool = False
    do_sample: bool = True
    interval_silence: int = Field(default=200, ge=0)
    max_text_tokens_per_segment: int = Field(default=120, ge=1)
    max_mel_tokens: int = Field(default=DEFAULT_MAX_MEL_TOKENS, ge=1)
    duration_factor: float = Field(default=1.0, ge=0.5, le=2.0)
    text_normalization: bool = True
    verbose: bool = False


class BatchFileResponse(BaseModel):
    status: str
    audio_paths: List[str]
    prompt_audio_path: str
    emo_audio_path: Optional[str]
    text_file: str
    sample_rate: Optional[int]


@app.post("/tts/synthesize", response_model=SynthesizeResponse)
def tts_synthesize(payload: SynthesizeRequest) -> SynthesizeResponse:
    if payload.emo_vector is not None and len(payload.emo_vector) != 8:
        raise HTTPException(status_code=422, detail="emo_vector must contain 8 float values.")
    try:
        language = _normalize_language(payload.lang)
        prompt_path = _resolve_audio_input(payload.prompt_wav_path)
        emo_prompt_path = _resolve_optional_audio(payload.emo_audio_prompt)
    except (FileNotFoundError, ValueError) as exc:
        status_code = 404 if isinstance(exc, FileNotFoundError) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    out_path = _build_output_path(payload.output_name)
    effective_use_emo_text = payload.use_emo_text or (payload.emo_text is not None)
    try:
        with _tts_lock:
            _run_inference(
                spk_audio_prompt=prompt_path,
                text=payload.text,
                output_path=out_path,
                lang=language,
                emo_audio_prompt=emo_prompt_path,
                emo_alpha=payload.emo_alpha,
                emo_vector=list(payload.emo_vector) if payload.emo_vector is not None else None,
                use_emo_text=effective_use_emo_text,
                emo_text=payload.emo_text,
                use_random=payload.use_random,
                interval_silence=payload.interval_silence,
                verbose=payload.verbose,
                max_text_tokens_per_segment=payload.max_text_tokens_per_segment,
                max_mel_tokens=payload.max_mel_tokens,
                raise_on_max_mel_tokens=True,
                do_sample=payload.do_sample,
                duration_factor=payload.duration_factor,
                text_normalization=payload.text_normalization,
            )
    except GenerationLengthExceededError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SynthesizeResponse(
        status="success",
        audio_path=out_path,
        prompt_audio_path=os.path.abspath(prompt_path),
        emo_audio_path=os.path.abspath(emo_prompt_path) if emo_prompt_path else None,
        sample_rate=_get_sample_rate(),
    )


@app.post("/tts/batch_file", response_model=BatchFileResponse)
def tts_batch_file(payload: BatchFileRequest) -> BatchFileResponse:
    if payload.emo_vector is not None and len(payload.emo_vector) != 8:
        raise HTTPException(status_code=422, detail="emo_vector must contain 8 float values.")
    try:
        language = _normalize_language(payload.lang)
        text_path = _resolve_text_file(payload.text_file)
        prompt_path = _resolve_audio_input(payload.prompt_wav_path)
        emo_prompt_path = _resolve_optional_audio(payload.emo_audio_prompt)
    except (FileNotFoundError, ValueError) as exc:
        status_code = 404 if isinstance(exc, FileNotFoundError) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    if payload.output_prefix:
        output_base = os.path.basename(payload.output_prefix)
        base_name = os.path.splitext(output_base)[0] or output_base
    else:
        base_name = os.path.splitext(os.path.basename(text_path))[0]

    output_paths: List[str] = []
    effective_use_emo_text = payload.use_emo_text or (payload.emo_text is not None)
    try:
        with _tts_lock:
            with open(text_path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    output_name = f"{base_name}_{line_no}.wav"
                    out_path = _build_output_path(output_name)
                    _run_inference(
                        spk_audio_prompt=prompt_path,
                        text=text,
                        output_path=out_path,
                        lang=language,
                        emo_audio_prompt=emo_prompt_path,
                        emo_alpha=payload.emo_alpha,
                        emo_vector=list(payload.emo_vector) if payload.emo_vector is not None else None,
                        use_emo_text=effective_use_emo_text,
                        emo_text=payload.emo_text,
                        use_random=payload.use_random,
                        interval_silence=payload.interval_silence,
                        verbose=payload.verbose,
                        max_text_tokens_per_segment=payload.max_text_tokens_per_segment,
                        max_mel_tokens=payload.max_mel_tokens,
                        raise_on_max_mel_tokens=True,
                        do_sample=payload.do_sample,
                        duration_factor=payload.duration_factor,
                        text_normalization=payload.text_normalization,
                    )
                    output_paths.append(out_path)
    except GenerationLengthExceededError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return BatchFileResponse(
        status="success",
        audio_paths=output_paths,
        prompt_audio_path=os.path.abspath(prompt_path),
        emo_audio_path=os.path.abspath(emo_prompt_path) if emo_prompt_path else None,
        text_file=text_path,
        sample_rate=_get_sample_rate(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
