import argparse
from pathlib import Path

from indextts.utils.runtime import configure_mps_environment

_mps_environment = configure_mps_environment()
if _mps_environment:
    print(
        ">> [System] macOS detected; MPS watermarks: "
        f"Low={_mps_environment['PYTORCH_MPS_LOW_WATERMARK_RATIO']}, "
        f"High={_mps_environment['PYTORCH_MPS_HIGH_WATERMARK_RATIO']}"
    )

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.device import clear_device_cache, device_type
from indextts.utils.exceptions import GenerationLengthExceededError


def _release_device_cache(tts: IndexTTS2) -> None:
    """Release tensors left by the previous utterance without dropping prompt caches."""
    cleared = clear_device_cache(tts.device, collect_garbage=True, synchronize=True)
    if cleared:
        tts.log_memory("after cleanup", synchronize=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch synthesize each line of a text file with IndexTTS2.5."
    )
    parser.add_argument("text_file", type=str, help="Path to input txt file (one sentence per line).")
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        required=True,
        help="Path to the speaker prompt audio (wav/mp3).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save wav files. Default: outputs",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to config file. Default: checkpoints/config.yaml",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints",
        help="Path to model directory. Default: checkpoints",
    )
    parser.add_argument("--bf16", action="store_true", help="Use BF16 if available.")
    parser.add_argument(
        "--lang",
        choices=("ZH", "EN", "JA", "ES", "AR"),
        default="ZH",
        help="Synthesis language. Default: ZH",
    )
    parser.add_argument(
        "--duration_factor",
        type=float,
        default=1.0,
        help="Speech duration multiplier from 0.5 to 2.0. Default: 1.0",
    )
    parser.add_argument(
        "--max_text_tokens_per_segment",
        type=int,
        default=50,
        help="Split long text into smaller segments. Default: 50",
    )
    parser.add_argument(
        "--max_mel_tokens",
        type=int,
        default=500,
        help="Maximum generated mel tokens per segment. Default: 500 (about 20 seconds)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries when generation reaches max_mel_tokens. Default: 2",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use random sampling. The safer default is deterministic beam search.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip output files that already exist (useful when resuming a batch).",
    )
    parser.add_argument(
        "--clear_device_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Clear the accelerator cache after each line. Default: enabled on MPS only.",
    )
    parser.add_argument(
        "--use_cuda_kernel",
        action="store_true",
        help="Enable BigVGAN CUDA kernel (CUDA only).",
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Enable DeepSpeed for GPT inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu, cuda:0, mps, xpu).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging in inference.",
    )
    parser.add_argument(
        "--memory_diagnostics",
        action="store_true",
        help=(
            "Synchronize accelerator stages and print detailed model, prompt, and "
            "inference memory checkpoints. Diagnostic timings include observer overhead."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_text_tokens_per_segment < 1:
        raise ValueError("--max_text_tokens_per_segment must be at least 1")
    if args.max_mel_tokens < 1:
        raise ValueError("--max_mel_tokens must be at least 1")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")

    text_path = Path(args.text_file)
    if not text_path.exists():
        raise FileNotFoundError(f"Input text file not found: {text_path}")

    voice_path = Path(args.voice)
    if not voice_path.exists():
        raise FileNotFoundError(f"Speaker prompt audio not found: {voice_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_bf16=args.bf16,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
        device=args.device,
        memory_diagnostics=args.memory_diagnostics,
    )
    clear_cache_after_each = (
        device_type(tts.device) == "mps"
        if args.clear_device_cache is None
        else args.clear_device_cache
    )

    base_name = text_path.stem
    completed = 0
    skipped = 0
    failed = []
    with text_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue

            output_name = f"{base_name}_{line_no}.wav"
            output_path = output_dir / output_name

            if args.skip_existing and output_path.is_file():
                print(f">> [{line_no}] skipped existing file: {output_path}")
                skipped += 1
                continue

            succeeded = False
            for attempt in range(args.retries + 1):
                # If deterministic generation reaches the safety limit, retry with
                # sampling. Further retries get a fresh random sample as well.
                do_sample = args.do_sample or attempt > 0
                mode = "sampling" if do_sample else "deterministic"
                print(
                    f">> [{line_no}] attempt {attempt + 1}/{args.retries + 1} "
                    f"({mode}): {text}"
                )
                limit_error = None
                result = None
                try:
                    result = tts.infer(
                        spk_audio_prompt=str(voice_path),
                        text=text,
                        output_path=str(output_path),
                        lang=args.lang,
                        duration_factor=args.duration_factor,
                        max_text_tokens_per_segment=args.max_text_tokens_per_segment,
                        max_mel_tokens=args.max_mel_tokens,
                        raise_on_max_mel_tokens=True,
                        do_sample=do_sample,
                        verbose=args.verbose,
                    )
                except GenerationLengthExceededError as error:
                    limit_error = error
                finally:
                    if clear_cache_after_each or limit_error is not None:
                        _release_device_cache(tts)

                succeeded = result is not None and output_path.is_file() and limit_error is None
                if succeeded:
                    completed += 1
                    break

                output_path.unlink(missing_ok=True)
                reason = str(limit_error) if limit_error is not None else "did not produce audio"
                print(f">> [{line_no}] {reason}; output removed before retry")

            if not succeeded:
                failed.append(line_no)
                print(f">> [{line_no}] failed after {args.retries + 1} attempts; continuing")

    print(
        f"Done. completed={completed}, skipped={skipped}, failed={len(failed)}. "
        f"Wav files saved under: {output_dir}"
    )
    if failed:
        print("Failed line numbers:", ", ".join(map(str, failed)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
