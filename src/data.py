import numpy as np
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms

# Load the MNIST training and test datasets
def load_mnist(data_dir="./data"):
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset

# Split a dataset approximately equally across clients
def split_iid(dataset, num_clients):
    base_size = len(dataset) // num_clients

    split_sizes = [base_size] * num_clients

    split_sizes[-1] += len(dataset) - sum(split_sizes)  # Add any remaining samples to the last client

    client_datasets = random_split(dataset, split_sizes)

    return client_datasets

# Split a dataset into non-IID subsets for each client using Dirichlet distribution
def split_non_iid(dataset, num_clients, alpha=0.5):
    labels = np.array(dataset.targets)
    client_indices = [[] for client_id in range(num_clients)]       # Create a list to hold the indices for each client
    for digit in range(10):
        class_indices = np.where(labels == digit)[0]                # Count the number of samples for each digit
        proportions = np.random.dirichlet([alpha] * num_clients)    # Calculate the proportions for each client using Dirichlet distribution
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)    # Calculate the split points for each client based on the proportions
        class_splits = np.split(class_indices, split_points)        # Split the digit samples among clients
        for client_id, indices in enumerate(class_splits):
            client_indices[client_id].extend(indices.tolist())      # Add the assigned samples to each client

        
    client_datasets = [Subset(dataset, indices)                     # Create a dataset subset for each client
        for indices in client_indices
    ]

    return client_datasets

def get_label_distribution(client_dataset):
    distribution = [0] * 10

    for _, label in client_dataset:
        distribution[label] += 1

    return distribution