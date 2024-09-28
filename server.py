import socket
import os

# Server settings
server_ip = '0.0.0.0'  # Listen on all available interfaces
server_port = 5001
save_dir = 'received_weights'  # Directory to save received weights
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Dictionary to store client and aggregated parameters for each round
parameters_storage = {
    'client_parameters': [],
    'aggregated_parameters': []
}

def save_weights(client_id, file_data):
    file_path = os.path.join(save_dir, f'client_{client_id}_weights.pt')
    with open(file_path, 'wb') as f:
        f.write(file_data)
    print(f"Saved weights for client {client_id} at {file_path}")

def aggregate_parameters():
    # Dummy aggregation logic
    # In a real-world FL scenario, you'd implement proper aggregation here (e.g., FedAvg)
    print("Aggregating client parameters...")
    # Simulate saving aggregated parameters
    aggregated_params_path = os.path.join(save_dir, f'aggregated_weights_round.pt')
    with open(aggregated_params_path, 'wb') as f:
        f.write(b'aggregated_parameters_data')  # Placeholder for actual data
    print(f"Aggregated parameters saved at {aggregated_params_path}")
    return aggregated_params_path

def send_aggregated_weights(client_socket, aggregated_file_path):
    # Send the aggregated weights back to the client
    with open(aggregated_file_path, 'rb') as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            client_socket.sendall(data)
    print(f"Aggregated weights sent back to client.")

def main():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(5)
    print(f"Server listening on {server_ip}:{server_port}")

    round_number = 1  # Round counter for federated learning rounds

    while True:
        print(f"\nWaiting for clients to send model weights for round {round_number}...")

        # Accept client connection
        client_socket, client_address = server_socket.accept()
        print(f"Connected by {client_address}")

        # Receive the file data from the client
        file_data = b''
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            file_data += data

        # Save received weights for this client
        client_id = client_address[1]  # Using port number as a proxy for client_id
        save_weights(client_id, file_data)

        # Add client parameters to storage (for aggregation later)
        parameters_storage['client_parameters'].append(file_data)

        # Aggregate parameters after receiving from all clients (assuming 1 client for now)
        if len(parameters_storage['client_parameters']) == 1:
            aggregated_file_path = aggregate_parameters()
            parameters_storage['aggregated_parameters'].append(b'aggregated_data')  # Simulate storage

            # Send the aggregated weights back to the client
            send_aggregated_weights(client_socket, aggregated_file_path)

            # Close the client connection after sending the file
            client_socket.close()

            round_number += 1  # Proceed to next round

if __name__ == "__main__":
    main()