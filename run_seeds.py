"""
Multi-seed comparison of aggregators.

For every seed we draw a NEW Dirichlet split, NEW Byzantine clients and a NEW initial
model, then train every method with and without a sign-flip attack. At the end we print
mean +/- std of the final accuracy across seeds, and a per-seed table that shows how the
share of samples held by the Byzantine clients relates to each method's accuracy.

The first seed is CONFIG["seed"], so it reproduces the earlier single-seed results
(with the same number of rounds).

Run from the repository root (next to run_experiment.py):
    python run_seeds.py                          # 3 seeds, 50 rounds per run
    python run_seeds.py --seeds 5 --rounds 50
    python run_seeds.py --seeds 1 --rounds 1     # quick check that everything works

Output: seeds_results.csv (rewritten after every seed, so partial results are kept).
"""
import argparse
import csv
import random
import time

import numpy as np
import torch

from run_experiment import CONFIG, run_training
from src.data import load_mnist, split_iid, split_non_iid, to_tensor_dataset


METHODS = ["mean", "geometric_median", "geometric_median_weighted", "huber"]
SCENARIOS = ["none", "sign_flip"]
SHORT = {"mean": "mean", "geometric_median": "gm", "geometric_median_weighted": "gm_weighted", "huber": "huber"}


def save_csv(rows, path="seeds_results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "byzantine", "byzantine_share", "scenario", "method",
                         "final_accuracy", "mean_last_10_rounds"])
        for r in rows:
            writer.writerow([r["seed"], r["byzantine"], r["byz_share"], r["scenario"], r["method"],
                             r["final_accuracy"], r["mean_last_10"]])


def summarize(rows):
    print("\n=== Final accuracy (%), mean +/- std across seeds ===")
    for scenario in SCENARIOS:
        print(f"  attack: {scenario}")
        for method in METHODS:
            values = [r["final_accuracy"] * 100 for r in rows
                      if r["scenario"] == scenario and r["method"] == method]
            if not values:
                continue
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            print(f"    {method:<26} {np.mean(values):6.2f} +/- {std:5.2f}   (n = {len(values)})")


def per_seed_table(rows):
    print("\n=== Per seed, attack: sign_flip, final accuracy (%) ===")
    header = " ".join(f"{SHORT.get(m, m):>12}" for m in METHODS)
    print(f"  {'seed':<5} {'Byzantine':<10} {'share':>6}  {header}")
    for seed in sorted({r["seed"] for r in rows}):
        seed_rows = [r for r in rows if r["seed"] == seed and r["scenario"] == "sign_flip"]
        if not seed_rows:
            continue
        byzantine = seed_rows[0]["byzantine"]
        share = seed_rows[0]["byz_share"] * 100
        accuracy = {r["method"]: r["final_accuracy"] * 100 for r in seed_rows}
        cells = " ".join(f"{accuracy.get(m, float('nan')):>12.2f}" for m in METHODS)
        print(f"  {seed:<5} {byzantine:<10} {share:>5.1f}%  {cells}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds (CONFIG seed, +1, +2, ...)")
    parser.add_argument("--rounds", type=int, default=50, help="training rounds per run")
    args = parser.parse_args()

    base_seed = CONFIG["seed"]
    train_dataset, test_dataset_raw = load_mnist()
    test_dataset = to_tensor_dataset(test_dataset_raw)

    rows = []
    runs_per_seed = len(SCENARIOS) * len(METHODS)

    for k in range(args.seeds):
        seed = base_seed + k
        cfg = dict(CONFIG, seed=seed, num_rounds=args.rounds)

        # New split, new Byzantine clients and new initial model for every seed.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if cfg["data_distribution"] == "non_iid":
            subsets = split_non_iid(train_dataset, num_clients=cfg["num_clients"], alpha=cfg["dirichlet_alpha"])
        else:
            subsets = split_iid(train_dataset, num_clients=cfg["num_clients"])

        client_datasets = [to_tensor_dataset(train_dataset, s) for s in subsets]
        client_sizes = [len(d) for d in client_datasets]

        # The Byzantine choice is always drawn from the seed (CONFIG["byzantine_clients"] is ignored here).
        byzantine_ids = random.Random(seed).sample(range(cfg["num_clients"]), cfg["num_byzantine"])
        byz_share = sum(client_sizes[i] for i in byzantine_ids) / sum(client_sizes)
        byzantine_label = ",".join(str(i + 1) for i in sorted(byzantine_ids))
        print(f"\n##### seed {seed} ({k + 1}/{args.seeds}): Byzantine clients {byzantine_label}, "
              f"{byz_share * 100:.1f}% of the samples #####")

        run_number = 0
        for scenario in SCENARIOS:
            for method in METHODS:
                run_number += 1
                print(f"\n=== seed {seed} [{run_number}/{runs_per_seed}] attack: {scenario} | {method} ===")
                start = time.time()
                history = run_training(method, scenario, cfg, client_datasets, client_sizes,
                                       test_dataset, byzantine_ids)
                print(f"    done in {time.time() - start:.0f}s")
                rows.append(dict(seed=seed, byzantine=byzantine_label, byz_share=byz_share,
                                 scenario=scenario, method=method, final_accuracy=history[-1],
                                 mean_last_10=float(np.mean(history[-10:]))))

        save_csv(rows)   # keep partial results if the run is interrupted

    summarize(rows)
    per_seed_table(rows)
    print("\nSaved seeds_results.csv")


if __name__ == "__main__":
    main()