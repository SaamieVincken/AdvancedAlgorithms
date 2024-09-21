import torch
import wandb
from torch import nn


def train_model(model, X, y, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X)  # Forward pass

        # Calculate Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(outputs - y))

        # Calculate R-squared
        r2 = r_squared(outputs, y)

        # Log metrics to W&B
        wandb.log({"mae": mae.item(), "r_squared": r2.item()})

        # Backpropagation
        mae.backward()  # You can use MAE for gradients if needed
        optimizer.step()  # Update weights


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions = model(X)

        mae = torch.mean(torch.abs(predictions - y))  # Calculate MAE
        r2 = r_squared(predictions, y)  # Calculate R-squared

    # Log test metrics to W&B
    wandb.log({"test_mae": mae.item(), "test_r_squared": r2.item()})
    return predictions.numpy()


# R-squared calculation
def r_squared(predictions, targets):
    ss_total = torch.sum((targets - torch.mean(targets)) ** 2)
    ss_residual = torch.sum((targets - predictions) ** 2)
    return 1 - (ss_residual / ss_total)
