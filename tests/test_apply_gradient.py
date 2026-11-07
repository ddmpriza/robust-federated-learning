import torch
from torch import nn

from src.gradient_aggregation import apply_gradient


# A simple model with two weights and one bias
model = nn.Linear(2, 1)

# Set known parameter values
with torch.no_grad():
    model.weight.copy_(torch.tensor([[2.0, -1.0]]))
    model.bias.copy_(torch.tensor([0.5]))

# Gradient order: weight[0], weight[1], bias
gradient = torch.tensor([0.5, -0.2, 0.1])

learning_rate = 0.1

# Apply global gradient update
apply_gradient(model, gradient, learning_rate)

print("Updated weights:", model.weight.data)
print("Updated bias:", model.bias.data)

# Expected values
expected_weights = torch.tensor([[1.95, -0.98]])
expected_bias = torch.tensor([0.49])

print(
    "Weights correct:",
    torch.allclose(model.weight, expected_weights)
)

print(
    "Bias correct:",
    torch.allclose(model.bias, expected_bias)
)