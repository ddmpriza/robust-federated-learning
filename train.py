import copy
import random
import torch

from src.data import load_mnist, split_iid
from src.model import MNISTModel
from src.client import train_client
from src.aggregation import fedavg
from src.evaluate import evaluate


def main():
    # Experiment settings
    seed = 42
    num_clients = 10
    num_rounds = 10
    local_epochs = 3
    batch_size = 32
    learning_rate = 0.01

    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, test_dataset = load_mnist()                              # Load MNIST

    client_datasets = split_iid(train_dataset, num_clients=num_clients)     # Split training data across clients

    global_model = MNISTModel()   
                                              # Create the initial global model
    accuracy_history = []

    # Federated training
    for round_id in range(num_rounds):

        print(f"\nFederated Round {round_id + 1}/{num_rounds}")

        local_models = []                                                           # Store the locally trained models
        client_sizes = []
        
        for client_id, client_dataset in enumerate(client_datasets):                # Train each client independently
            print(f"\nTraining Client {client_id + 1}/{num_clients}")
            
            local_model = copy.deepcopy(global_model)                               # Every client starts from the same global model

            local_model = train_client(                                             # Local training
                model=local_model,
                dataset=client_dataset,
                epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            local_models.append(local_model)
            client_sizes.append(len(client_dataset))

        global_model = fedavg(local_models=local_models, client_sizes=client_sizes)     # Aggregate the local models to form a new global model

        accuracy = evaluate(model=global_model, dataset=test_dataset)                   # Evaluate the new global model
        accuracy_history.append(accuracy)

        print(f"\nGlobal model accuracy: {accuracy * 100:.2f}%")

        print("\nFederated round completed.")


if __name__ == "__main__":
    main()