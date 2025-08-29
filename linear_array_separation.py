"""Linear_Array_Separation
Original file is located at
https://colab.research.google.com/drive/1psa0obFUYSaa16Q6WaA2pm71Td09hhsr
"""
pip install numpy scipy soundfile pyroomacoustics
!pip install torch torchaudio
!pip install speechbrain
!pip install -q git+https://github.com/openai/whisper.git
!sudo apt update && sudo apt install -y ffmpeg
!pip install -q jiwer torchaudio soundfile

import soundfile as sf
import numpy as np
import pyroomacoustics as pra
import scipy.signal
from numpy import load
import os
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as Separator
import whisper
from jiwer import wer
from scipy.io import wavfile
import matplotlib.pyplot as plt
from ipywidgets import VBox, Output
from IPython.display import display, Audio

data1, sr1 = sf.read("/content/1272-128104-0009.flac")
data2, sr2 = sf.read("/content/1673-143396-0009.flac")

sf.write("source1.wav", data1, sr1)
sf.write("source2.wav", data2, sr2)

# Parameters
mic_count = 8
mic_spacing = 0.08  # 8 cm
room_dim = [5, 5]  # 5m x 5m room
fs = 16000
source_angles = [30, 70]  # Degrees
source_distance = 1.5  # Distance from array center to sources
rir_duration = 0.4  # seconds
snr_db = 30

# Load Clean LibriSpeech Files
source1_audio, _ = sf.read('source1.wav')
source2_audio, _ = sf.read('source2.wav')

# Ensure both sources have same length for mixing
min_len = min(len(source1_audio), len(source2_audio))
source1_audio = source1_audio[:min_len]
source2_audio = source2_audio[:min_len]

#  Microphone Array Setup
# Center of mic array
array_center = np.array([room_dim[0]/2, room_dim[1]/2])

# Linear array along x-axis
mic_positions = np.zeros((2, mic_count))
mic_positions[0, :] = np.linspace(-mic_spacing*(mic_count-1)/2, mic_spacing*(mic_count-1)/2, mic_count)
mic_positions += array_center.reshape(2,1)

# Source Positions
source_positions = []
for angle_deg in source_angles:
    angle_rad = np.deg2rad(angle_deg)
    x = array_center[0] + source_distance * np.cos(angle_rad)
    y = array_center[1] + source_distance * np.sin(angle_rad)
    source_positions.append([x, y])

#  Room Setup
room = pra.ShoeBox(room_dim, fs=fs, max_order=10, absorption=0.4)

# Add sources
room.add_source(source_positions[0], signal=source1_audio)
room.add_source(source_positions[1], signal=source2_audio)

# Add mic array
mic_array_2d = mic_positions
room.add_microphone_array(mic_array_2d)

#  Simulate and Mix
room.simulate()

# Get mixed signals (shape: mic_count x samples)
mixed_signals = room.mic_array.signals

#  Save Mixed Signals
for i in range(mic_count):
    sf.write(f'mic_{i+1}.wav', mixed_signals[i], fs)

np.save("mixed_matrix.npy", mixed_signals)
print(" Saved as mixed_matrix.npy")
print(mixed_signals.shape)
matrix = np.load("mixed_matrix.npy")
data = load("mixed_matrix.npy")

# ----------Seperation----------------
# ------------------ config ------------------
MIXED_NPY = "mixed_matrix.npy"     # shape: (M, T)
TARGET_SR = 8000                   # SepFormer expects 8 kHz
ARRAY_SR  = 16000                  # your array simulation fs
C = 343.0

# ------------------ helpers ------------------
def to_float(x):
    x = x.astype(np.float32)
    return x / (np.max(np.abs(x)) + 1e-9)

def resample_1d(x, orig, new):
    t = torch.tensor(x).float()
    if orig != new:
        t = torchaudio.functional.resample(t, orig_freq=orig, new_freq=new)
    return t.numpy()

def run_sepformer(mono, sr_in=16000, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x8 = resample_1d(mono, sr_in, TARGET_SR)
    x8 = np.squeeze(x8)                  # ensure 1D
    wav = torch.tensor(x8).float().unsqueeze(0)  # [B,T]
    sep = Separator.from_hparams(
        source="speechbrain/sepformer-whamr",
        savedir="pretrained_models/sepformer-whamr",
        run_opts={"device": device}
    )
    with torch.no_grad():
        out = sep.separate_batch(wav.to(device))  # [B,T,2]
    s1 = out[0, :, 0].cpu().numpy()
    s2 = out[0, :, 1].cpu().numpy()
    return s1, s2, TARGET_SR

# ------------------ pipeline ------------------
# 1) load array mixture
assert os.path.exists(MIXED_NPY), f"{MIXED_NPY} not found"
mix_arr = np.load(MIXED_NPY)  # (M,T)
mix_arr = to_float(mix_arr)

# 2) pick center mic (baseline)
center_idx = mix_arr.shape[0] // 2
mono_center = mix_arr[center_idx]

# 3) run SepFormer
s1, s2, sr_sep = run_sepformer(mono_center, sr_in=ARRAY_SR)

# 4) save results
sf.write("est_source1.wav", s1, sr_sep, subtype="PCM_16")
sf.write("est_source2.wav", s2, sr_sep, subtype="PCM_16")
print(f" Separation complete → est_source1.wav / est_source2.wav @ {sr_sep} Hz")

# -----------------------------------------
mixed_matrix = np.load("mixed_matrix.npy")
mic1_waveform = mixed_matrix[0]
mic1_waveform = np.clip(mic1_waveform, -1.0, 1.0)
mic1_int16 = (mic1_waveform * 32767).astype(np.int16)
TARGET_SAMPLE_RATE = 16000
wavfile.write("mixed.wav", TARGET_SAMPLE_RATE, mic1_int16)
print(" Saved mic 1 waveform as mixed_mic1.wav")

#-----------------------Visulisation of outputs-----------------------

# Load all files with actual sample rate
mix, sr_mix = sf.read("mixed.wav")
src1, sr1 = sf.read("est_source1.wav")
src2, sr2 = sf.read("est_source2.wav")

# Create time axes for plotting
t_mix = np.linspace(0, len(mix)/sr_mix, len(mix))
t1 = np.linspace(0, len(src1)/sr1, len(src1))
t2 = np.linspace(0, len(src2)/sr2, len(src2))

out_mix = Output()
out_s1 = Output()
out_s2 = Output()

with out_mix:
    print(" Mixed Signal")
    plt.figure(figsize=(10, 2))
    plt.plot(t_mix, mix, color='black')
    plt.title("Mixed Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    display(Audio(mix, rate=sr_mix))

with out_s1:
    print(" Separated Source 1")
    plt.figure(figsize=(10, 2))
    plt.plot(t1, src1, color='green')
    plt.title("Separated Source 1")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    display(Audio(src1, rate=sr1))  # Correct sample rate

with out_s2:
    print(" Separated Source 2")
    plt.figure(figsize=(10, 2))
    plt.plot(t2, src2, color='blue')
    plt.title("Separated Source 2")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    display(Audio(src2, rate=sr2))  # Correct sample rate

display(VBox([out_mix, out_s1, out_s2]))

# ------------------------ Transcribtion And wer Calculation -------------------------

# ------------------------ LOAD WHISPER MODEL -------------------------
model = whisper.load_model("small")

# ------------------------ FILE PATHS -------------------------
files = {
    "source1": "source1.wav",
    "source2": "source2.wav",
    "est_source1": "est_source1.wav",
    "est_source2": "est_source2.wav"
}

# ------------------------ TRANSCRIBE -------------------------
transcripts = {}
for label, path in files.items():
    if not os.path.exists(path):
        print(f" File not found: {path}")
        continue

    print(f"\n Transcribing {path}:")
    display(Audio(path))
    result = model.transcribe(path)
    transcripts[label] = result["text"].strip()
    print(f" {label} transcript: {transcripts[label]}")

# ------------------------ PERMUTATION-INVARIANT WER -------------------------
ref1, ref2 = transcripts["source1"], transcripts["source2"]
hyp1, hyp2 = transcripts["est_source1"], transcripts["est_source2"]

# Case 1: no swap
wer_case1 = wer(ref1.lower(), hyp1.lower()) + wer(ref2.lower(), hyp2.lower())
# Case 2: swapped
wer_case2 = wer(ref1.lower(), hyp2.lower()) + wer(ref2.lower(), hyp1.lower())

# ------------------------ ALIGN & SAVE -------------------------
TARGET_SR = 16000

if wer_case1 <= wer_case2:
    print("\n Best alignment: est_source1 ↔ source1, est_source2 ↔ source2")
    print(f"WER source1 vs est_source1: {wer(ref1.lower(), hyp1.lower()):.2%}")
    print(f"WER source2 vs est_source2: {wer(ref2.lower(), hyp2.lower()):.2%}")

    # Save aligned files
    audio1, sr1 = torchaudio.load("est_source1.wav")
    audio2, sr2 = torchaudio.load("est_source2.wav")
else:
    print("\n Best alignment: est_source1 ↔ source2, est_source2 ↔ source1 (swapped)")
    print(f"WER source1 vs est_source2: {wer(ref1.lower(), hyp2.lower()):.2%}")
    print(f"WER source2 vs est_source1: {wer(ref2.lower(), hyp1.lower()):.2%}")

    # Swap assignments
    audio1, sr1 = torchaudio.load("est_source2.wav")
    audio2, sr2 = torchaudio.load("est_source1.wav")

# Resample both to 16kHz and save
if sr1 != TARGET_SR:
    audio1 = torchaudio.functional.resample(audio1, sr1, TARGET_SR)
if sr2 != TARGET_SR:
    audio2 = torchaudio.functional.resample(audio2, sr2, TARGET_SR)

sf.write("est_source1_aligned.wav", audio1.squeeze().numpy(), TARGET_SR, subtype='PCM_16')
sf.write("est_source2_aligned.wav", audio2.squeeze().numpy(), TARGET_SR, subtype='PCM_16')

print(" Saved aligned outputs: est_source1_aligned.wav / est_source2_aligned.wav at 16kHz")
#-------------wer output---------------------------
1. WER source1 vs est_source2: 16.67%

2. WER source2 vs est_source1: 8.77%
