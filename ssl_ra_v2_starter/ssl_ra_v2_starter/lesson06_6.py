r"""Lesson 06-6: Learn DOA directly from two-microphone IPD features.

Standalone training script for a synthetic, direct-path-only classroom experiment.
Uses the same microphone geometry and delay simulator as main.py.

Run from the ssl_gcc_phat_starter folder:
    .\.venv\Scripts\python.exe lesson06_6.py

Outputs: outputs_ipd/ipd_doa_model.pth, test_results.csv, prediction_scatter.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, stft
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from main import FS, C, D, simulate_pair, gcc_phat_stft, tau_to_angle

# ----------------------------------------------
# Experimental settings
# ----------------------------------------------
OUT = Path(__file__).resolve().parent / "outputs_ipd"
N_TRAIN = 800
N_VAL = 160
N_TEST = 160
N_SAMPLES = 5120              # 0.32 seconds at 16 kHz
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.002
N_PER_SEG = 512
HOP = 256
N_FFT = 512
FREQ_LIMIT = C / (2 * D)  # 857.5 Hz: avoid spatial phase aliasing


def create_source(rng: np.random.Generator, sos: np.ndarray) -> np.ndarray:
    """Fresh random broadband source for EVERY sample; no train/test source reuse."""
    white = rng.normal(size=N_SAMPLES)
    src = sosfiltfilt(sos, white)
    # Slowly changing amplitude envelope mimics nonstationarity.
    t = np.arange(N_SAMPLES) / FS
    rate = rng.uniform(1.5, 5.0)
    env = 0.35 + 0.65 * np.sin(np.pi * rate * t + rng.uniform(0, 2 * np.pi))**2
    src *= env
    return (src / (np.max(np.abs(src)) + 1e-9)).astype(np.float64)


def extract_ipd(mic1: np.ndarray, mic2: np.ndarray) -> np.ndarray:
    """Compute cos(IPD) and sin(IPD) at each selected STFT frequency."""
    f, _, z1 = stft(mic1, fs=FS, nperseg=N_PER_SEG,
                     noverlap=N_PER_SEG-HOP, nfft=N_FFT,
                     boundary=None, padded=False)
    _, _, z2 = stft(mic2, fs=FS, nperseg=N_PER_SEG,
                     noverlap=N_PER_SEG-HOP, nfft=N_FFT,
                     boundary=None, padded=False)
    # IPD convention is phi2 - phi1, matching the original GCC-PHAT script.
    cross = np.mean(z2 * np.conj(z1), axis=1)
    bins = (f >= 125.0) & (f <= FREQ_LIMIT)
    unit_phase = cross[bins] / (np.abs(cross[bins]) + 1e-10)
    # Shape: (number_of_frequencies, 2) -> (2 * number_of_frequencies,)
    return np.stack((unit_phase.real, unit_phase.imag), axis=1).reshape(-1).astype(np.float32)


def make_dataset(count: int, seed: int, keep_audio: bool = False):
    rng = np.random.default_rng(seed)
    sos = butter(4, [120, 3000], btype="bandpass", fs=FS, output="sos")
    xs, ys, angles, audio = [], [], [], []
    for _ in range(count):
        theta = rng.uniform(-70.0, 70.0)
        snr = rng.uniform(8.0, 30.0)
        src = create_source(rng, sos)
        mic1, mic2, _ = simulate_pair(theta, src, snr_db=snr,
                                       reverb=False,
                                       seed=int(rng.integers(0, 2**31)))
        xs.append(extract_ipd(mic1, mic2))
        # Target is sin(theta) in [-1, 1], NOT the theoretical TDOA input.
        ys.append(np.sin(np.deg2rad(theta)))
        angles.append(theta)
        if keep_audio:
            audio.append((mic1, mic2))
    return (np.stack(xs), np.asarray(ys, dtype=np.float32).reshape(-1, 1),
            np.asarray(angles), audio)


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(1)  # predictable CPU use on beginner laptops

    print("Generating independent synthetic train/test audio ...")
    x_train, y_train, _, _ = make_dataset(N_TRAIN, seed=42)
    x_val, y_val, theta_val, _ = make_dataset(N_VAL, seed=2025)
    x_test, y_test, theta_test, test_audio = make_dataset(N_TEST, seed=2026, keep_audio=True)
    print(f"Train features: {x_train.shape}; Train targets: {y_train.shape}")
    print(f"Validation features: {x_val.shape}; Validation targets: {y_val.shape}")
    print(f"Test features:  {x_test.shape}; Test targets:  {y_test.shape}")
    print(f"Input features per example: {x_train.shape[1]} (cos/sin IPD)")

    loader = DataLoader(TensorDataset(torch.from_numpy(x_train),
                                     torch.from_numpy(y_train)),
                        batch_size=BATCH_SIZE, shuffle=True)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Tanh(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    model.train()

    for epoch in range(1, EPOCHS + 1):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        if epoch in {1, 20, 50, 100}:
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(x_val)).numpy().ravel()
            doa = np.rad2deg(np.arcsin(np.clip(pred, -1.0, 1.0)))
            mae = np.mean(np.abs(doa - theta_val))
            print(f"Epoch {epoch:3d}/{EPOCHS} | Validation MAE: {mae:.2f} deg")
            model.train()

    model.eval()
    with torch.no_grad():
        raw_pred = model(torch.from_numpy(x_test)).numpy().ravel()
    nn_deg = np.rad2deg(np.arcsin(np.clip(raw_pred, -1.0, 1.0)))

    # Fair comparison: the same held-out two-channel audio goes to GCC-PHAT.
    gcc_deg = []
    for mic1, mic2 in test_audio:
        tau_hat, *_ = gcc_phat_stft(mic1, mic2, fs=FS)
        gcc_deg.append(tau_to_angle(tau_hat))
    gcc_deg = np.asarray(gcc_deg)

    mae_nn = float(np.mean(np.abs(nn_deg - theta_test)))
    mae_gcc = float(np.mean(np.abs(gcc_deg - theta_test)))
    print(f"Held-out tests: {N_TEST} (never used for training or validation)")
    print(f"Neural IPD MAE:  {mae_nn:.2f} deg")
    print(f"GCC-PHAT MAE:   {mae_gcc:.2f} deg")
    print("Scenario: synthetic direct path, 8-30 dB SNR, no reverberation")

    torch.save({"state_dict": model.state_dict(),
                "sample_rate": FS, "mic_spacing_m": D,
                "n_fft": N_FFT, "window": N_PER_SEG,
                "feature_count": x_train.shape[1]},
               OUT / "ipd_doa_model.pth")

    with (OUT / "test_results.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true_doa_deg", "nn_doa_deg", "gcc_doa_deg",
                         "nn_abs_error_deg", "gcc_abs_error_deg"])
        for t, n, g in zip(theta_test, nn_deg, gcc_deg):
            writer.writerow([f"{t:.5f}", f"{n:.5f}", f"{g:.5f}",
                             f"{abs(n-t):.5f}", f"{abs(g-t):.5f}"])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(theta_test, gcc_deg, s=10, alpha=.55, label=f"GCC-PHAT (MAE {mae_gcc:.2f} deg)")
    ax.scatter(theta_test, nn_deg, s=10, alpha=.55, label=f"IPD network (MAE {mae_nn:.2f} deg)")
    ax.plot([-75, 75], [-75, 75], "k--", lw=1, label="Ideal")
    ax.set(xlabel="Ground-truth DOA (deg)", ylabel="Estimated DOA (deg)",
           title="Synthetic two-microphone SSL: held-out test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "prediction_scatter.png", dpi=160)
    plt.close(fig)
    print(f"Saved results: {OUT.resolve()}")


if __name__ == "__main__":
    main()
