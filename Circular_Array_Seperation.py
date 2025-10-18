# =====================================================
# Import Libraries
# =====================================================
import numpy as np
import scipy.io as sio
import soundfile as sf
from scipy.signal import resample_poly
import torch
from speechbrain.inference.separation import SepformerSeparation as separator
import whisper
from jiwer import wer
from IPython.display import Audio, display

# =====================================================
# Load and Mix Audio from .mat Files
# =====================================================
mat_data4 = sio.loadmat('/content/audio_smir_single.mat')
mat_data3 = sio.loadmat('/content/audio_smir_single2.mat')

audio_smir4 = mat_data4['audio_smir']
audio_smir_data3 = mat_data3['audio_smir']

# Ensure same length
L = min(audio_smir4.shape[1], audio_smir_data3.shape[1])
mix_smir = audio_smir4[:, :L] + audio_smir_data3[:, :L]

# Save mixed result
sio.savemat('audio_smir_mix.mat', {'mix_smir': mix_smir})
print("Audio files loaded, mixed, and saved successfully!")

# =====================================================
# Convert Mixed Audio to WAV (Mono)
# =====================================================
mat = sio.loadmat("audio_smir_mix.mat")
mix_smir = mat["mix_smir"]  # shape: (Mics, L)

# Option A: average all microphones to mono
mix_mono = mix_smir.mean(axis=0)

# Option B: select single mic channel
# mix_mono = mix_smir[0, :]

sf.write("mixture.wav", mix_mono, 16000)  # Save at 16kHz
print("Saved 16kHz mixture.wav")

# =====================================================
# Downsample to 8kHz
# =====================================================
TARGET_SR = 8000
mixture_8k = resample_poly(mix_mono, TARGET_SR, 16000)
sf.write("mixture_8k.wav", mixture_8k, TARGET_SR)
print("Saved 8kHz mixture_8k.wav")

# =====================================================
# Load and Run Speech Separation Model (SepFormer)
# =====================================================
sep_model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

mix_tensor = torch.tensor(mixture_8k).unsqueeze(0).float().to(sep_model.device)
est_sources = sep_model.separate_batch(mix_tensor)

out1 = est_sources[0, :, 0].cpu().numpy()
out2 = est_sources[0, :, 1].cpu().numpy()

sf.write("est_source1.wav", out1, TARGET_SR)
sf.write("est_source2.wav", out2, TARGET_SR)
print("Separated sources saved as est_source1.wav and est_source2.wav")

# =====================================================
# Playback Audio for Verification
# =====================================================
print("Original Mixture")
display(Audio(mixture_8k, rate=TARGET_SR))

print("Estimated Source 1")
display(Audio(out1, rate=TARGET_SR))

print("Estimated Source 2")
display(Audio(out2, rate=TARGET_SR))

# =====================================================
# Speech Recognition (ASR) and WER Evaluation
# =====================================================
asr_model = whisper.load_model("medium")

# Load clean reference sources
data1, sr1 = sf.read("/content/1673-143396-0005.flac")
data2, sr2 = sf.read("/content/1462-170138-0005.flac")

sf.write("source1.wav", data1, sr1)
sf.write("source2.wav", data2, sr2)

# Helper function for transcription
def transcribe_audio(filepath):
    result = asr_model.transcribe(filepath, language="en")
    return result["text"]

# Transcribe clean and separated sources
ref_text1 = transcribe_audio("source1.wav")
ref_text2 = transcribe_audio("source2.wav")
hyp1 = transcribe_audio("est_source1.wav")
hyp2 = transcribe_audio("est_source2.wav")

print("Clean Source1:", ref_text1)
print("Clean Source2:", ref_text2)
print("Source1 transcription:", hyp1)
print("Source2 transcription:", hyp2)

# Compute Word Error Rate (WER)
wer1 = wer(ref_text2, hyp1)
wer2 = wer(ref_text1, hyp2)

print(f"WER for Source1: {wer1:.2f}")
print(f"WER for Source2: {wer2:.2f}")
