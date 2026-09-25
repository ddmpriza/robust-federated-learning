from torch.utils.data import random_split
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

if __name__ == "__main__":

    train_dataset, test_dataset = load_mnist()

    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    num_clients = 10

    client_datasets = split_iid(
        train_dataset,
        num_clients
    )

    print("\nNumber of clients:", len(client_datasets))

    for i, client_dataset in enumerate(client_datasets):
        print(f"Client {i}: {len(client_dataset)} samples")