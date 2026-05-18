import argparse
import os
import platform
import sys
if platform.system() == "Darwin":
    # 允许 MPS 回退到 CPU (防止显存不足直接报错)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # 设置内存垃圾回收的水位线 (60% - 80%)，防止吃满物理内存导致 Swap 爆炸
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.6"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
    print(f">> [System] 检测到 macOS 环境，已自动应用 MPS 内存水位限制 (Low=0.6, High=0.8)")
from pathlib import Path

from indextts.infer_v2 import IndexTTS2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch synthesize each line of a text file with IndexTTS2."
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
    parser.add_argument("--fp16", action="store_true", help="Use FP16 if available.")
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
    return parser.parse_args()


def main():
    args = parse_args()

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
        use_fp16=args.fp16,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
        device=args.device,
    )

    base_name = text_path.stem
    with text_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue

            output_name = f"{base_name}_{line_no}.wav"
            output_path = output_dir / output_name

            tts.infer(
                spk_audio_prompt=str(voice_path),
                text=text,
                output_path=str(output_path),
                # use_emo_text=True,
                verbose=args.verbose,
            )

    print(f"Done. Wav files saved under: {output_dir}")


if __name__ == "__main__":
    main()
