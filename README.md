#  Speech Separation & Recognition Using Linear and Circular Microphone Arrays

## Project Overview
This project presents an **end-to-end speech separation and recognition pipeline** combining classical and deep learning techniques.  
It demonstrates **how microphone array geometry** (linear vs. circular) affects source separation quality and downstream **Automatic Speech Recognition (ASR)** accuracy.  

The work integrates **array signal processing**, **deep learning separation (SepFormer)**, and **speech recognition (Whisper)** to build a reproducible framework for experimental and research use.

---

##  Key Contributions
- Simulated **multi-channel mixtures** using both Linear and Circular microphone arrays.  
- Implemented **MVDR beamformer** for classical array processing and comparison baseline.  
- Built neural separation pipeline using **SpeechBrain’s SepFormer** model.  
- Integrated **OpenAI Whisper** for transcribing clean, mixed, and separated signals.  
- Evaluated performance using **Word Error Rate (WER)** via `jiwer`.  
- Complete modular workflow:  
> **Dataset Generation → Mixing → Beamforming → Separation → Transcription → WER Evaluation → Visualization**

---


##  MVDR Beamforming Formulation

The MVDR (Minimum Variance Distortionless Response) beamformer minimizes the output power while preserving the desired signal direction:

$$
\min_{\mathbf{w}} \ \mathbf{w}^H \mathbf{R} \mathbf{w}
\quad \text{s.t. } \mathbf{w}^H \mathbf{d} = 1
$$

where:  
- $\mathbf{R}$ — covariance matrix of microphone signals  
- $\mathbf{d}$ — steering vector toward the desired direction  

The **closed-form solution** is given by:

$$
\mathbf{w}_{\text{MVDR}} =
\frac{\mathbf{R}^{-1}\mathbf{d}}
{\mathbf{d}^H \mathbf{R}^{-1}\mathbf{d}}
$$


This provides **distortionless filtering** towards the target and **suppresses interference/noise** from other directions.

---

##  Data Generation

###  Linear Array Simulation (Python)
- Simulated a **linear microphone array** using `pyroomacoustics`.  
- Each microphone records the superposition of multiple sources and room reflections.  
- Used **LibriSpeech clean utterances** as inputs.  
- Generated noisy/reverberant mixtures using known array geometry.

### Example (Linear Array)
```python
import numpy as np, pyroomacoustics as pra, soundfile as sf

mic_count = 8
mic_spacing = 0.08  # 8 cm
room = pra.ShoeBox([5,5], fs=16000, max_order=10)
array_center = np.array([2.5,2.5])
mic_positions = np.vstack([
    np.linspace(-mic_spacing*(mic_count-1)/2, mic_spacing*(mic_count-1)/2, mic_count) + array_center[0],
    np.ones(mic_count) * array_center[1]
])
room.add_microphone_array(mic_positions)

# Add sources
room.add_source([3.5, 2.0], signal=sf.read('source1.wav')[0])
room.add_source([2.0, 3.5], signal=sf.read('source2.wav')[0])

room.simulate()
sf.write('mixed.wav', room.mic_array.signals.mean(axis=0), 16000)
```
Circular Microphone Array (MATLAB – SMIR Generator)
```python
load('em32micpos.mat');
sphRadius = 0.042; % 4.2 cm
procFs = 16000;

for phi_src = 0:5:85
    for theta_src = 0:5:85
        [h_tmp,~,~] = smir_generator(c, procFs, sphLocation, pos_src, L, beta, sphType, sphRadius, mic, N_harm, nsample, K, order);
        for ind = 1:32
            audio_smir(ind,:) = conv2(h_tmp(:,ind), s_temp);
        end
        audio_sig = awgn(audio_smir, snr, 'measured');
        save(sprintf('audio_smir_azi%d_ele%d.mat', phi_src, theta_src), 'audio_sig');
    end
end
```
---
##  Experimental Workflow Summary
Step	Description	Tools Used
1. Dataset Generation	Simulate multi-mic recordings: PyRoomAcoustics / SMIR	MATLAB, Python
2. Beamforming (MVDR)	Classical spatial filtering:	NumPy, SciPy
3. Neural Separation:	SepFormer-based reverberant separation using SpeechBrain
4. Transcription:	Speech-to-text via Whisper using OpenAI Whisper
5. Evaluation: WER computation	using JiWER
6. Visualization:	Time-domain and spectrograms using	Matplotlib

---
## Sample Results

| Configuration       | Array Type       | Avg WER Before | Avg WER After | Improvement |
|----------------------|------------------|----------------|----------------|--------------|
| Linear (8 mics)      | Straight array   | 0.92           | 0.48           | 47.8%        |
| Circular (32 mics)   | Rigid sphere     | 0.91           | 0.39           | 57.1%        |

