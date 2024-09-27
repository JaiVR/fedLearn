# import flwr as fl
# from typing import List, Tuple
# import torch

# # Define aggregation strategy for Flower server
# def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
#     """Compute weighted average of client metrics."""
#     accuracy_sum = sum([num_examples * m["accuracy"] for num_examples, m in metrics])
#     num_examples_total = sum([num_examples for num_examples, m in metrics])
#     return {"accuracy": accuracy_sum / num_examples_total}

# # Start Flower server with the chosen strategy
# if __name__ == "__main__":
#     # Use FedAvg (Federated Averaging) strategy
#     strategy = fl.server.strategy.FedAvg(
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_available_clients= 1 # Ensure at least 2 clients are available
#     )
    
#     # Start the server on the specified address and port
#     fl.server.start_server(
#         server_address="192.168.0.103:5001",  # Server's IP and port
#         strategy=strategy
#     )
import flwr as fl

def main():
    # Define strategy with the number of rounds
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=1,  # Minimum number of clients required for a round            # Number of rounds
    )

    # Start Flower server with the defined strategy
    server_address = "0.0.0.0:8080"  # Listen on all interfaces (allowing remote connections)
    print(f"Starting Flower server at {server_address}")

    fl.server.start_server(
        server_address=server_address,
        strategy=strategy
    )

if __name__ == "__main__":
    main()
