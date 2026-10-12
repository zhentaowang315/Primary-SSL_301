# Two-microphone sound-source localization: GCC-PHAT and an IPD neural baseline

An educational **synthetic-data** sound-source localization (SSL) project with a fixed two-microphone geometry. It demonstrates signal processing (STFT, GCC-PHAT, TDOA, DOA) and a small supervised neural network trained on *observed* inter-channel phase differences (IPD).

## Setup (Windows PowerShell)

Requires Python 3.10+.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
# Classical two-channel simulation
.\.venv\Scripts\python.exe main.py --angle 30 --snr 10 --out outputs_30

# Multiple-angle GCC-PHAT baseline
.\.venv\Scripts\python.exe evaluate.py --out outputs_eval

# Train a small neural model from IPD features and compare on held-out audio
.\.venv\Scripts\python.exe lesson06_6.py
```

## Method

- `main.py`: synthesize two microphone waveforms, compute STFT cross-spectra, use GCC-PHAT to estimate TDOA, and invert the two-mic far-field geometry to estimate DOA.
- `lesson06_6.py`: generate independent synthetic train/validation/test audio (800/160/160 examples). For each example, calculate the IPD between microphones, encoded as sine/cosine pairs at **24 low-frequency bins** (<857.5 Hz), and predict `sin(DOA)` with an MLP. Training uses Adam and MSE.
- The IPD network **does not receive ground-truth TDOA or DOA** as an input. The angle is only the supervised target.
- A GCC-PHAT baseline is computed on the **same held-out test audio**.
- Output CSV columns contain ground truth, both predictions, and both absolute errors; a scatter plot illustrates the results.

## Experimental conditions and limitations

- Simulation only; two-microphone spacing = 0.20 m, sample rate = 16 kHz, sound speed = 343 m/s, source azimuth = −70° to +70° (broadside = 0°, right = positive).
- In the IPD-neural experiment, independent random source signals and noise realizations are generated for train/validation/test. SNR is sampled from 8–30 dB, with direct-path propagation only, **no reverberation**. The training and test scenes share the same simulator.
- Simple interpolation models fractional delays; this is not high-fidelity room acoustics. Real recordings may have synchronization errors, room reflections, microphone mismatch, and nonstationary interference.
- Any better neural results in this simplified test do **not** establish superiority over GCC-PHAT in real scenes.
- This is a learning baseline, not a new algorithm or published research result.

## Suggested next steps

1. Add independent room impulse responses and real recordings.
2. Evaluate robustness over SNR, DOA, and reverberation strength with repeated seeds.
3. Add reproducible training configs, an untouched test set, and experiment descriptions.

## Files

- `main.py` — GCC-PHAT baseline and synthetic two-channel WAV generator.
- `evaluate.py` — multiple-angle evaluation.
- `lesson06_6.py` — PyTorch IPD-to-DOA model and evaluation.
- `requirements.txt` — dependencies.
- `.gitignore` — prevents uploading environments and generated files by accident.
