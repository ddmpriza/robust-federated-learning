import copy

from src.data import load_mnist, split_iid
from src.model import MNISTModel
from src.client import train_client
from src.aggregation import fedavg
from src.evaluate import evaluate


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

    global_model = fedavg(local_models=local_models, client_sizes=client_sizes)  # Aggregate the local models to form a new global model

    
    accuracy = evaluate(model=global_model, dataset=test_dataset)               # Evaluate the new global model

    print(f"\nGlobal model accuracy: {accuracy * 100:.2f}%")

    print("\nFederated round completed.")


if __name__ == "__main__":
    main()