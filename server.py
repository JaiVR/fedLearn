import flwr as fl
from typing import List, Tuple

# Define the aggregation strategy
class SaveAndAggregate(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException]
    ) -> Tuple[List[float], dict]:
        # Call the super class's aggregation function
        aggregated_parameters, aggregation_info = super().aggregate_fit(rnd, results, failures)

        # Save aggregated parameters to disk
        if aggregated_parameters is not None:
            aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            path = f"server_aggregated_round_{rnd}.pt"
            with open(path, "wb") as f:
                torch.save(aggregated_weights, f)
            print(f"Aggregated model saved at {path}")

        return aggregated_parameters, aggregation_info

# Start Flower server
def start_server():
    strategy = SaveAndAggregate(
        min_fit_clients=1,  # Minimum number of clients to participate in a round of training
        min_available_clients=1,  # Minimum number of total clients that need to be connected to the server
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8000",  # Server IP and port
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),  # Configure the number of FL rounds
    )

if __name__ == "__main__":
    start_server()