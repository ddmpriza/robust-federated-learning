"""
Sweep of the Huber threshold constant `huber_c`.

The Huber aggregator interpolates between two aggregators we already have:
  small huber_c -> behaves like the geometric median
  large huber_c -> behaves like the weighted mean
This script trains with several values of huber_c (with and without a sign-flip attack)
and plots the final accuracy, using mean and geometric median as reference lines.

Run from the repository root (next to run_experiment.py):
    python run_huber_sweep.py               # default: 50 rounds per run
    python run_huber_sweep.py --rounds 10   # quick check that everything works

Outputs: huber_sweep.csv and huber_sweep.png (the plot needs matplotlib).
"""
import argparse
import csv
import random
import time

import numpy as np
import torch

from run_experiment import CONFIG, to_tensor_dataset, run_training
from src.data import load_mnist, split_iid, split_non_iid


C_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0, 5.0]
SCENARIOS = ["none", "sign_flip"]
REFERENCE_METHODS = ["mean", "geometric_median"]


def save_csv(results, path="huber_sweep.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "method", "huber_c", "final_accuracy", "mean_last_10_rounds"])
        for scenario in SCENARIOS:
            for method in REFERENCE_METHODS:
                history = results[(scenario, method)]
                writer.writerow([scenario, method, "", history[-1], float(np.mean(history[-10:]))])
            for c in C_VALUES:
                history = results[(scenario, f"huber_c={c}")]
                writer.writerow([scenario, "huber", c, history[-1], float(np.mean(history[-10:]))])
    print(f"Saved {path}")


def save_plot(results, path="huber_sweep.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed -> skipping the plot (pip install matplotlib)")
        return

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(5.5 * len(SCENARIOS), 4), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        finals = [results[(scenario, f"huber_c={c}")][-1] for c in C_VALUES]
        ax.plot(C_VALUES, finals, marker="o", label="huber")
        ax.axhline(results[(scenario, "mean")][-1], linestyle="--", color="gray", label="mean")
        ax.axhline(results[(scenario, "geometric_median")][-1], linestyle=":", color="black",
                   label="geometric median")
        ax.set_xscale("log")
        ax.set_xlabel("huber_c")
        ax.set_title(f"attack: {scenario}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("final test accuracy")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50, help="training rounds per run")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["num_rounds"] = args.rounds

    # Same data setup (and seeds) as run_experiment.py, so the results are comparable.
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    train_dataset, test_dataset_raw = load_mnist()
    if cfg["data_distribution"] == "non_iid":
        subsets = split_non_iid(train_dataset, num_clients=cfg["num_clients"], alpha=cfg["dirichlet_alpha"])
    else:
        subsets = split_iid(train_dataset, num_clients=cfg["num_clients"])

    client_datasets = [to_tensor_dataset(train_dataset, s) for s in subsets]
    client_sizes = [len(d) for d in client_datasets]
    test_dataset = to_tensor_dataset(test_dataset_raw)
    byzantine_ids = random.Random(cfg["seed"]).sample(range(cfg["num_clients"]), cfg["num_byzantine"])
    print(f"Byzantine clients (attack scenarios only): {sorted(i + 1 for i in byzantine_ids)}")

    results = {}
    total_runs = len(SCENARIOS) * (len(REFERENCE_METHODS) + len(C_VALUES))
    run_number = 0

    for scenario in SCENARIOS:
        runs = [(method, method, cfg) for method in REFERENCE_METHODS]
        runs += [("huber", f"huber_c={c}", dict(cfg, huber_c=c)) for c in C_VALUES]

        for method, label, run_cfg in runs:
            run_number += 1
            print(f"\n=== [{run_number}/{total_runs}] attack: {scenario} | {label} ===")
            start = time.time()
            results[(scenario, label)] = run_training(
                method, scenario, run_cfg, client_datasets, client_sizes, test_dataset, byzantine_ids
            )
            print(f"    done in {time.time() - start:.0f}s")

    print("\nFinal accuracy:")
    for scenario in SCENARIOS:
        print(f"  attack: {scenario}")
        for label in REFERENCE_METHODS + [f"huber_c={c}" for c in C_VALUES]:
            print(f"    {label:<18} {results[(scenario, label)][-1] * 100:6.2f}%")

    save_csv(results)
    save_plot(results)


if __name__ == "__main__":
    main()