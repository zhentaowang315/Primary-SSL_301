"""评估多组已知角度的 DOA 误差，复用 main.py 的整条算法链。"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import signal_source, simulate_pair, gcc_phat_stft, tau_to_angle


def main():
    parser = argparse.ArgumentParser(description="Evaluate DOA estimation over multiple angles")
    parser.add_argument("--snr", type=float, default=30)
    parser.add_argument("--reverb", action="store_true")
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = signal_source()
    angles = [-70, -50, -30, -15, 0, 15, 30, 50, 70]
    rows = []
    for theta in angles:
        mic1, mic2, tau_true = simulate_pair(theta, source, snr_db=args.snr,
                                              reverb=args.reverb, seed=42)
        tau_pred, *_ = gcc_phat_stft(mic1, mic2)
        theta_pred = tau_to_angle(tau_pred)
        error = theta_pred - theta
        rows.append([theta, theta_pred, error, abs(error), tau_true*1e6, tau_pred*1e6])
        print(f"True {theta:+4d} deg -> Estimated {theta_pred:+7.2f} deg | Abs error={abs(error):.2f} deg")
    mae = float(np.mean([r[3] for r in rows]))
    print(f"Mean absolute error (MAE) = {mae:.2f} deg")
    with (out / "angle_evaluation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_deg", "pred_deg", "signed_error_deg", "abs_error_deg",
                         "tdoa_true_us", "tdoa_pred_us"])
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(angles, angles, linestyle="--", label="Perfect estimate")
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label="GCC-PHAT result")
    ax.set(xlabel="True DOA (deg)", ylabel="Estimated DOA (deg)",
           title=f"DOA estimation, MAE = {mae:.2f} deg")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_angle_evaluation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
