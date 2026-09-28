import copy
import torch

# Aggregate local models using Federated Averaging (FedAvg)
# client_sizes: number of samples in each client's dataset
def fedavg(local_models, client_sizes):
    global_model = copy.deepcopy(local_models[0])       # Copy the first local model

    global_state = global_model.state_dict()            # Retrieve the dictionary with the parameters of the global model
    total_samples = sum(client_sizes)

    
    for key in global_state:                                        # Sepearate aggregation for each parameter in the model
        global_state[key] = torch.zeros_like(global_state[key])     # Start the weighted sum from zero

        for model, client_size in zip(local_models, client_sizes):  # Add the weighted parameter from each client
            weight = client_size / total_samples

            global_state[key] += (weight * model.state_dict()[key])

    # Load the aggregated parameters into the global model
    global_model.load_state_dict(global_state)

    return global_model