from collections import OrderedDict

import torch
from centralized import load_datasets, train, test, train_yolov5
from server import load_model
from yolov5.train import run

import flwr as fl

import sys

from yolov5.utils.dataloaders import LoadImagesAndLabels

partition_id = int(sys.argv[1])
EPOCHS = 1

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def save_model(model, path=f"client_model{partition_id}.pt"):
    torch.save(model, path)
    print(f"Model saved to {path}")

net = load_model()
trainloader, _, testloader = load_datasets(partition_id)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def fit(self, parameters, config):
        set_parameters(net, parameters)
        run(net)
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}    

fl.client.start_numpy_client(
    server_address="192.168.0.105:8000",
    client=FlowerClient()
)