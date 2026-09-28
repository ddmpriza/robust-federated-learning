import copy

from src.data import load_mnist, split_iid
from src.model import MNISTModel
from src.client import train_client
from src.aggregation import fedavg


def main():

    # Load MNIST
    train_dataset, test_dataset = load_mnist()

    # Split training data across clients
    num_clients = 10
    client_datasets = split_iid(
        train_dataset,
        num_clients=num_clients
    )

    # Create the initial global model
    global_model = MNISTModel()

    # Store the locally trained models
    local_models = []
    client_sizes = []

    # Train each client independently
    for client_id, client_dataset in enumerate(client_datasets):
        print(f"\nTraining Client {client_id + 1}/{num_clients}")
        
        local_model = copy.deepcopy(global_model)               # Every client starts from the same global model

        local_model = train_client(                             # Local training
            model=local_model,
            dataset=client_dataset,
            epochs=3,
            batch_size=32,
            learning_rate=0.01
        )

        local_models.append(local_model)
        client_sizes.append(len(client_dataset))

    global_model = fedavg(                                      # Aggregate all local models using FedAvg
        local_models=local_models,
        client_sizes=client_sizes
    )

    print("\nFederated round completed.")


if __name__ == "__main__":
    main()