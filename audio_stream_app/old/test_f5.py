import torch
import torchaudio
from f5_tts.api import F5TTS

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize F5TTS model
try:
    # Initialize with default model (F5TTS_v1_Base)
    tts = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file="f5_tts_repo/ckptsX/en_us_cmudict-0.02_f5_ft_phone/model_ckpt_steps_120000.ckpt",
        use_ema=True
    )

    # Reference audio and text for voice cloning
    ref_audio = "path/to/reference.wav"  # Add path to your reference audio
    ref_text = "Reference text matching the audio"  # Add text matching reference audio
    
    # Text to synthesize
    gen_text = "Hello, this is a test of the F5 TTS system."

    # Generate speech
    print(f"Synthesizing speech for: {gen_text}")
    wav, sr, spec = tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave="f5_tts_output.wav"  # Output will be saved here
    )
    
    print(f"Saved audio to: f5_tts_output.wav")

except Exception as e:
    print(f"Error during synthesis: {e}")
