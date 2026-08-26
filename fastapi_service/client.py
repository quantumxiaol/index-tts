import argparse
import json
import os
import sys
from typing import Any

import httpx


DEFAULT_BASE_URL = os.getenv("INDEXTTS_API_URL", "http://127.0.0.1:8000")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple httpx client for the IndexTTS FastAPI service")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="FastAPI service base URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    synthesize = subparsers.add_parser("synthesize", help="Call POST /tts/synthesize")
    _add_common_tts_args(synthesize)
    synthesize.add_argument("--text", required=True, help="Text to synthesize")
    synthesize.add_argument("--output-name", help="Optional output file name on the server")

    batch_file = subparsers.add_parser("batch-file", help="Call POST /tts/batch_file")
    _add_common_tts_args(batch_file)
    batch_file.add_argument("--text-file", required=True, help="Path to input text file on the server or local machine")
    batch_file.add_argument("--output-prefix", help="Optional output prefix on the server")

    return parser


def _add_common_tts_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-wav-path", required=True, help="Local path or URL of the speaker prompt audio")
    parser.add_argument(
        "--lang",
        choices=("ZH", "EN", "JA", "ES", "AR"),
        default="ZH",
        help="Synthesis language (default: ZH)",
    )
    parser.add_argument("--emo-audio-prompt", help="Local path or URL of the emotion reference audio")
    parser.add_argument("--emo-alpha", type=float, default=1.0, help="Weight of the emotion reference")
    parser.add_argument(
        "--emo-vector",
        nargs=8,
        type=float,
        metavar=("HAPPY", "ANGRY", "SAD", "AFRAID", "DISGUSTED", "MELANCHOLIC", "SURPRISED", "CALM"),
        help="8-dim emotion vector",
    )
    parser.add_argument("--use-emo-text", action="store_true", help="Enable emotion-from-text mode")
    parser.add_argument("--emo-text", help="Optional emotion description text")
    parser.add_argument(
        "--use-random",
        action="store_true",
        help="Use random emotion reference embeddings",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable GPT sampling",
    )
    parser.add_argument("--interval-silence", type=int, default=200, help="Silence between segments in milliseconds")
    parser.add_argument(
        "--max-text-tokens-per-segment",
        type=int,
        default=120,
        help="Maximum text tokens per synthesis segment",
    )
    parser.add_argument(
        "--max-mel-tokens",
        type=int,
        default=None,
        help="Maximum generated mel tokens per segment (default: server setting)",
    )
    parser.add_argument(
        "--duration-factor",
        type=float,
        default=1.0,
        help="Speech duration multiplier from 0.5 to 2.0",
    )
    parser.add_argument(
        "--text-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable text normalization",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose inference on the server")


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt_wav_path": args.prompt_wav_path,
        "lang": args.lang,
        "emo_audio_prompt": args.emo_audio_prompt,
        "emo_alpha": args.emo_alpha,
        "emo_vector": list(args.emo_vector) if args.emo_vector is not None else None,
        "use_emo_text": args.use_emo_text,
        "emo_text": args.emo_text,
        "use_random": args.use_random,
        "do_sample": args.do_sample,
        "interval_silence": args.interval_silence,
        "max_text_tokens_per_segment": args.max_text_tokens_per_segment,
        "duration_factor": args.duration_factor,
        "text_normalization": args.text_normalization,
        "verbose": args.verbose,
    }
    if args.max_mel_tokens is not None:
        payload["max_mel_tokens"] = args.max_mel_tokens

    if args.command == "synthesize":
        payload["text"] = args.text
        payload["output_name"] = args.output_name
    elif args.command == "batch-file":
        payload["text_file"] = args.text_file
        payload["output_prefix"] = args.output_prefix

    return payload


def _endpoint_for(command: str) -> str:
    if command == "synthesize":
        return "/tts/synthesize"
    if command == "batch-file":
        return "/tts/batch_file"
    raise ValueError(f"Unsupported command: {command}")


def _post_json(base_url: str, endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        message = detail or str(exc)
        raise SystemExit(f"Request failed with status {exc.response.status_code}: {message}") from exc
    except httpx.HTTPError as exc:
        raise SystemExit(f"Request failed: {exc}") from exc

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Server returned non-JSON response: {response.text}") from exc


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = _build_payload(args)
    result = _post_json(
        base_url=args.base_url,
        endpoint=_endpoint_for(args.command),
        payload=payload,
        timeout=args.timeout,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
