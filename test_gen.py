from indextts.infer_v2_5 import IndexTTS2

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "训练员，你看这个训练计划怎么样？"
tts.infer(spk_audio_prompt="Admire_Vega.mp3", text=text, lang="ZH", output_path="gen.wav", verbose=True)
