from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "训练员，你看这个训练计划怎么样？"
tts.infer(spk_audio_prompt='Admire_Vega.mp3', text=text, output_path="gen.wav", verbose=True)