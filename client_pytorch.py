import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# Setup for parsing command-line arguments
parser = argparse.ArgumentParser(description="Flower Embedded Devices")

# Add arguments for server address and client ID
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help="gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)

# Add argument to specify if MNIST dataset should be used (useful for lower memory devices)
parser.add_argument(
    "--mnist",
    action="store_true",
    help="If you use Raspberry Pi Zero clients (which just have 512MB of RAM) use MNIST",
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Define number of clients
NUM_CLIENTS = 50


# Define a simple CNN model for MNIST dataset
class Net(nn.Module):
    """Simple CNN model for MNIST."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Training function
def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()


# Testing function
def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# Dataset preparation function
def prepare_dataset(use_mnist: bool):
    """Get MNIST/CIFAR-10 and return client partitions and global testset."""
    if use_mnist:
        fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
        img_key = "image"
        norm = Normalize((0.1307,), (0.3081,))
    else:
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
        img_key = "img"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pytorch_transforms = Compose([ToTensor(), norm])

    def apply_transforms(batch):
        """Apply transforms to each data partition."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    # Create train and validation sets for each client
    trainsets, validsets = [], []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        # Split data into 90% train, 10% test
        partition = partition.train_test_split(test_size=0.1, seed=42)
        partition = partition.with_transform(apply_transforms)
        trainsets.append(partition["train"])
        validsets.append(partition["test"])

    # Create global test set
    testset = fds.load_split("test")
    testset = testset.with_transform(apply_transforms)
    return trainsets, validsets, testset


# Define the Flower client class
class FlowerClient(fl.client.NumPyClient):
    """A Flower client that trains a model for CIFAR-10 or MNIST."""

    def __init__(self, trainset, valset, use_mnist, cid):
        self.trainset = trainset
        self.valset = valset
        self.cid = cid  # Client ID
        # Instantiate model
        self.model = Net() if use_mnist else mobilenet_v3_small(num_classes=10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Training function."""
        print("Client sampled for fit()")
        self.set_parameters(parameters)

        # Hyperparameters from server config
        batch, epochs = config["batch_size"], config["epochs"]

        # Construct DataLoader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # Train model
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)

        # Return updated model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluation function."""
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)

        # Create DataLoader for validation
        valloader = DataLoader(self.valset, batch_size=64)

        # Evaluate model
        loss, accuracy = test(self.model, valloader, device=self.device)

        # Save model after each round
        model_path = f"client_{self.cid}_round_{config.get('round', 0)}.pt"
        torch.save(self.model, model_path)
        print(f"Model saved at {model_path}")

        # Return evaluation metrics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def main():
    """Main function to start the Flower client."""
    args = parser.parse_args()
    print(args)

    # Ensure client ID is valid
    assert args.cid < NUM_CLIENTS

    # Decide dataset based on client memory
    use_mnist = args.mnist
    # Partition dataset for each client
    trainsets, valsets, _ = prepare_dataset(use_mnist)

    # Start Flower client with respective data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid], use_mnist=use_mnist, cid=args.cid
        ).to_client(),
    )


if __name__ == "__main__":
    main()
