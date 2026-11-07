import random
import torch
import numpy as np

from src.data import load_mnist, split_iid, split_non_iid, get_label_distribution
from src.model import MNISTModel
from src.client import compute_client_gradient
from src.gradient_aggregation import gradient_mean, apply_gradient
from src.evaluate import evaluate


def main():
    # Experiment settings
    seed = 42
    num_clients = 10
    num_rounds = 10
    batch_size = 32
    learning_rate = 0.1
    data_distribution = "non_iid"  # Options: "iid" or "non_iid"
    dirichlet_alpha = 0.5

    # Reproducibility
    random.seed(seed)           # Python randomness
    np.random.seed(seed)        # Dirichlet distribution randomness
    torch.manual_seed(seed)     # PyTorch randomness

    train_dataset, test_dataset = load_mnist()                              # Load MNIST

    if data_distribution == "non_iid":
        client_datasets = split_non_iid(train_dataset, num_clients=num_clients, alpha=dirichlet_alpha)
    else:
        client_datasets = split_iid(train_dataset, num_clients=num_clients)

    # Display client label distributions
    for client_id, client_dataset in enumerate(client_datasets):
        distribution = get_label_distribution(client_dataset)

        print(f"Client {client_id + 1}: "f"{len(client_dataset)} samples - "f"{distribution}")

    global_model = MNISTModel()               # Create the initial global model               
    accuracy_history = []

    # Federated training
    for round_id in range(num_rounds):

        print(f"\nFederated Round {round_id + 1}/{num_rounds}")

        client_gradients = []                     # Store the gradients computed by each client                   
        client_sizes = []

        for client_id, client_dataset in enumerate(client_datasets):                # Train each client independently
            print(f"\nTraining Client {client_id + 1}/{num_clients}")
            
            #local_model = copy.deepcopy(global_model)                               # Every client starts from the same global model

            gradient = compute_client_gradient(
                model=global_model,
                dataset=client_dataset,
                batch_size=batch_size
            )
            client_gradients.append(gradient)                       # Store the computed gradient for this client
            client_sizes.append(len(client_dataset))                # Store the number of samples in this client's dataset

        gradients = torch.stack(client_gradients)                   # Stack client gradients into a matrix
        # Aggregate client gradients
        aggregated_gradient = gradient_mean(
            gradients=gradients,
            client_sizes=torch.tensor(client_sizes)
        )

        # Update global model
        global_model = apply_gradient(
            model=global_model,
            gradient=aggregated_gradient,
            learning_rate=learning_rate
        )

            
        accuracy = evaluate(model=global_model, dataset=test_dataset)
        accuracy_history.append(accuracy)
        print(f"\nGlobal model accuracy: {accuracy * 100:.2f}%")
        print("Federated round completed.")

    print("\nTraining completed.")
    print("Accuracy history:")

    for round_id, accuracy in enumerate(accuracy_history):
        print(f"Round {round_id + 1}: {accuracy * 100:.2f}%")

    print("\nTraining completed.")
    print("Accuracy history:")

    for round_id, accuracy in enumerate(accuracy_history):
        print(f"Round {round_id + 1}: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()