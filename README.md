# Byzantine-Robust Federated Learning

An experimental PyTorch implementation of federated learning with a focus on **robust gradient aggregation** in the presence of Byzantine clients and heterogeneous (non-IID) data.

The project is being developed as part of a master's thesis. The main reference is *A Huber Loss Minimization Approach to Byzantine Robust Federated Learning* (Zhao, Yu, Wan, [arXiv:2308.12581](https://arxiv.org/abs/2308.12581)); the geometric median aggregator follows *Robust Aggregation for Federated Learning* (Pillutla, Kakade, Harchaoui, [arXiv:1912.13445](https://arxiv.org/abs/1912.13445)).

## Idea

In each round every client sends a vector (here: its full local gradient) to the server. The server has to combine them into a single update. Plain (weighted) averaging is sensitive to malicious clients, so we study robust location estimators for this step.

The Huber aggregator minimizes `sum_i n_i * Huber_T(||s - g_i||)` over `s`, where `g_i` is the gradient of client `i` and `n_i` its number of samples. The threshold `T` interpolates between two known aggregators:

- `T -> infinity`: the weighted mean (FedAvg-style averaging).
- `T -> 0`: the geometric median.

## Repository structure

```
src/
  data.py                  MNIST loading, IID and Dirichlet non-IID splits
  model.py                 MLP for MNIST
  client.py                local training (train_client) and full local gradient (compute_client_gradient)
  aggregation.py           FedAvg over model weights
  gradient_aggregation.py  gradient aggregators and apply_gradient
  evaluate.py              test accuracy
tests/                     pytest tests for gradients, aggregation, geometric median, Huber
train.py                   FedAvg baseline (local training, weight averaging)
train_gradients.py         gradient-based training loop (weighted mean aggregation)
run_experiment.py          compares aggregators with and without an attack
```

## Aggregators (`src/gradient_aggregation.py`)

| Function | Description |
|---|---|
| `gradient_mean` | Mean weighted by client dataset size |
| `geometric_median` | Modified Weiszfeld iteration (handles gradients that coincide with the current center); unweighted |
| `huber_aggregation` | Huber aggregation weighted by client sizes, iteratively reweighted; supports a threshold per client |
| `huber_aggregation_paper` | Unweighted Huber update as written in Section 5 of Zhao et al. |

## Setting

`train_gradients.py` and `run_experiment.py` follow Algorithm 1 of Zhao et al.: each client computes its **full local gradient** at the current global model, the server aggregates the gradients and takes one step `w <- w - lr * g`. This is different from `train.py`, where clients train locally for several epochs and the server averages model weights.

## Experiments

```bash
pip install -r requirements.txt
pip install matplotlib      # optional, only for the plot
python run_experiment.py
```

`run_experiment.py` trains with `mean`, `geometric_median` and `huber`, with and without a sign-flip attack (a fraction of clients send `-5 * gradient`), and writes `results.csv` (and `results.png` if matplotlib is installed). Settings are in the `CONFIG` dictionary at the top of the file.

**Huber threshold.** The threshold is set per round as `T = huber_c * median distance of the client gradients from their coordinate-wise median`. This is a heuristic of this project, not taken from the paper; the paper uses a fixed `T` (and `T_i = T_0 + M / sqrt(n_i)` for unbalanced data).

### Preliminary result

10 clients, Dirichlet split with alpha = 0.5, 2 Byzantine clients (about 25% of all samples), 10 rounds, one seed. Test accuracy after the last round:

| | no attack | sign-flip |
|---|---|---|
| mean | 81.1% | 10.3% |
| geometric median | 79.6% | 61.2% |
| Huber (`huber_c = 1`) | 80.3% | 58.4% |

This is a quick sanity check, not a conclusion: a single seed, 10 rounds, and one choice of Byzantine clients. The difference between the geometric median and Huber is within what could be noise.

## Planned work

- Longer runs (100+ rounds) and several seeds.
- Sweep of the Huber threshold, from geometric-median-like to mean-like behavior.
- More attacks and baselines (coordinate-wise median, trimmed mean, Krum).
- Comparison under IID and non-IID data, with different fractions of Byzantine clients.

## Status

Under active development. Implementations and the experimental setup may change as the research progresses.