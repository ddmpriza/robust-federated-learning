# Byzantine-Robust Federated Learning

This repository contains an experimental implementation of federated learning methods, with a focus on robust gradient aggregation in the presence of Byzantine clients.

The project is being developed as part of a master's thesis and aims to investigate how different aggregation strategies behave under heterogeneous data distributions and adversarial conditions.

## Overview

In federated learning, multiple clients collaboratively train a model without sharing their local training data. Instead, clients communicate model updates or gradients to a central server, which aggregates the received information to update the global model.

Standard aggregation methods can be sensitive to corrupted or malicious client contributions. This project explores alternative aggregation strategies designed to reduce the influence of such contributions.

## Current Implementation

The current implementation includes:

- A basic Federated Averaging (FedAvg) baseline.
- IID and non-IID data partitioning across clients.
- Client-side gradient computation and server-side gradient aggregation.
- Arithmetic mean aggregation, weighted by client dataset size.
- An initial implementation of geometric median aggregation using the Weiszfeld algorithm.
- Basic numerical tests for gradient computation, aggregation, and model updates.

The geometric median implementation is currently undergoing additional validation, particularly for numerical edge cases.

## Experimental Setup

The initial experiments use:

- **Framework:** PyTorch
- **Dataset:** MNIST
- **Model:** Multilayer Perceptron (MLP)
- **Data distribution:** IID and non-IID partitions

The experimental setup is intended to support comparisons between aggregation methods under different client data distributions and Byzantine attack scenarios.

## Planned Work

The next stages of development include:

- Further validation of the geometric median implementation.
- Implementation of Huber loss-based gradient aggregation.
- Introduction of Byzantine client behaviors.
- Comparative evaluation of aggregation methods under IID and non-IID settings.
- Analysis of model performance and robustness under different experimental conditions.

## Project Status

This project is under active development. The implementations and experimental setup may change as the research progresses.