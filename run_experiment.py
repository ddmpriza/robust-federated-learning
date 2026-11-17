"""
First experiment: how do different aggregators behave with and without
Byzantine clients?

Setting (follows Algorithm 1 of Zhao et al.):
  - every client sends its full local gradient,
  - the server aggregates the gradients and takes one step  w <- w - lr * g.

Run from the repository root:
    python run_experiment.py

Outputs: results.csv (accuracy per round) and results.png (plot, needs matplotlib).
"""
import csv
import random
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.data import load_mnist, split_iid, split_non_iid
from src.model import MNISTModel
from src.client import compute_client_gradient
from src.gradient_aggregation import  gradient_mean, apply_gradient, geometric_median, huber_aggregation
from src.evaluate import evaluate


# Settings
CONFIG = dict(
    seed=42,
    num_clients=10,
    num_rounds=10,                  # the paper uses 200-500 iterations
    learning_rate=0.5,
    data_distribution="non_iid",    # "iid" or "non_iid"
    dirichlet_alpha=0.5,
    grad_batch_size=2048,           # only affects speed, not the result (full-batch gradient)
    eval_batch_size=2000,
    num_byzantine=2,                # epsilon = 2/10 = 0.2
    attack_scale=5.0,               # sign_flip sends  -scale * gradient
    huber_c=1.0                    # Huber threshold T = huber_c * median distance (see robust_threshold)
)

METHODS = ["mean", "geometric_median", "huber"]
SCENARIOS = ["none", "sign_flip"]   # "gaussian" is also available 


# Convert (a subset of) torchvision MNIST to in-memory tensors (much faster than applying the transform sample by sample in every round).
def to_tensor_dataset(base_dataset, subset=None):
    if subset is None:
        idx = torch.arange(len(base_dataset))
    else:
        idx = torch.as_tensor(subset.indices, dtype=torch.long)
        idx = torch.as_tensor(subset.indices, dtype=torch.long)
    x = base_dataset.data[idx].float().div(255.0).unsqueeze(1)   # [n, 1, 28, 28], same scaling as ToTensor
    y = base_dataset.targets[idx].long()
    return TensorDataset(x, y)


def apply_attack(gradients, byzantine_ids, attack, scale):
    attacked = gradients.clone()
    if attack == "none" or len(byzantine_ids) == 0:
        return attacked

    for i in byzantine_ids:
        if attack == "sign_flip":
            attacked[i] = -scale * gradients[i]
        elif attack == "gaussian":
            attacked[i] = gradients[i] + scale * gradients[i].std() * torch.randn_like(gradients[i])
        else:
            raise ValueError(f"Unknown attack: {attack}")
    return attacked


# Data-driven Huber threshold (a simple heuristic of ours, NOT from the paper):
# T = c * median distance of the client gradients from their coordinate-wise median.
# A fixed T is hard to choose because gradient norms change during training.
def robust_threshold(gradients, c):
    center = gradients.median(dim=0).values
    distances = torch.linalg.vector_norm(gradients - center, dim=1)
    return max((c * distances.median()).item(), 1e-8)

# Returns (aggregated_gradient, huber_threshold_or_None).
def aggregate(method, gradients, client_sizes, cfg):
    if method == "mean":
        return gradient_mean(gradients, client_sizes), None
    if method == "geometric_median":
        return geometric_median(gradients), None
    if method == "huber":
        threshold = robust_threshold(gradients, cfg["huber_c"])
        return huber_aggregation(gradients, client_sizes, thresholds=threshold), threshold
    raise ValueError(f"Unknown aggregation method: {method}")


def run_training(method, attack, cfg, client_datasets, client_sizes, test_dataset, byzantine_ids):
    torch.manual_seed(cfg["seed"])          # every run starts from the same initial model
    model = MNISTModel()
    sizes = torch.tensor(client_sizes)
    history = []

    for round_id in range(cfg["num_rounds"]):
        gradients = torch.stack([
            compute_client_gradient(model, dataset, batch_size=cfg["grad_batch_size"])
            for dataset in client_datasets
        ])
        gradients = apply_attack(gradients, byzantine_ids, attack, cfg["attack_scale"])

        aggregated, threshold = aggregate(method, gradients, sizes, cfg)
        model = apply_gradient(model, aggregated, cfg["learning_rate"])

        accuracy = evaluate(model, test_dataset, batch_size=cfg["eval_batch_size"])
        history.append(accuracy)

        if (round_id + 1) % 10 == 0 or round_id == 0:
            extra = f"  (Huber T = {threshold:.4f})" if threshold is not None else ""
            print(f"    round {round_id + 1:>3}/{cfg['num_rounds']}: accuracy {accuracy * 100:6.2f}%{extra}")

    return history


def save_csv(results, path="results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "method", "round", "accuracy"])
        for (scenario, method), history in results.items():
            for round_id, accuracy in enumerate(history, start=1):
                writer.writerow([scenario, method, round_id, accuracy])
    print(f"Saved {path}")


def save_plot(results, path="results.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed -> skipping the plot (pip install matplotlib)")
        return

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6 * len(SCENARIOS), 4), sharey=True)
    if len(SCENARIOS) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, SCENARIOS):
        for method in METHODS:
            history = results[(scenario, method)]
            ax.plot(range(1, len(history) + 1), history, label=method)
        ax.set_title(f"attack: {scenario}")
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


# Main
def main():
    cfg = CONFIG

    # Reproducibility
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])        # Dirichlet split
    torch.manual_seed(cfg["seed"])     # random_split / model init

    train_dataset, test_dataset_raw = load_mnist()

    if cfg["data_distribution"] == "non_iid":
        subsets = split_non_iid(train_dataset, num_clients=cfg["num_clients"], alpha=cfg["dirichlet_alpha"])
    else:
        subsets = split_iid(train_dataset, num_clients=cfg["num_clients"])

    client_datasets = [to_tensor_dataset(train_dataset, s) for s in subsets]
    client_sizes = [len(d) for d in client_datasets]
    test_dataset = to_tensor_dataset(test_dataset_raw)

    print("Client label distributions:")
    for client_id, dataset in enumerate(client_datasets):
        counts = torch.bincount(dataset.tensors[1], minlength=10).tolist()
        print(f"  Client {client_id + 1}: {len(dataset)} samples - {counts}")

    byzantine_ids = random.Random(cfg["seed"]).sample(range(cfg["num_clients"]), cfg["num_byzantine"])
    print(f"\nByzantine clients (used only in attack scenarios): {sorted(i + 1 for i in byzantine_ids)}")

    results = {}
    for scenario in SCENARIOS:
        for method in METHODS:
            print(f"\n=== attack: {scenario} | aggregator: {method} ===")
            start = time.time()
            results[(scenario, method)] = run_training(
                method, scenario, cfg, client_datasets, client_sizes, test_dataset, byzantine_ids
            )
            print(f"    done in {time.time() - start:.0f}s")

    print("\nFinal accuracy:")
    for (scenario, method), history in results.items():
        print(f"  {scenario:<10} {method:<17} {history[-1] * 100:6.2f}%")

    save_csv(results)
    save_plot(results)


if __name__ == "__main__":
    main()