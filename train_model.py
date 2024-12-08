import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np

df = pd.read_csv(r'./processed_data.csv')
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

X = df[["humidity", "wind_speed"]].values
y = df["temperature"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32,64)
        self.fc4 = nn.Linear(64,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SimpleNN(input_dim=2)  
criterion = nn.MSELoss()
lr = 0.002
optimizer = optim.Adam(model.parameters(), lr=lr)

epochs = 10
batch_size = 32
num_batches = len(X_train_tensor) // batch_size

with mlflow.start_run() as run:
    mlflow.log_param("model", "BaseLine NN")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", lr)
    print(f"Run ID: {run.info.run_id}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size

            # Get the batch
            X_batch = X_train_tensor[batch_start:batch_end]
            y_batch = y_train_tensor[batch_start:batch_end]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log the epoch loss
        mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.numpy().flatten()
        mse = mean_squared_error(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("mse", mse)
    X_test_numpy = X_test_tensor.numpy()


    mlflow.pytorch.log_model(model, "Deep NN")

    print(f"Logged model with MSE: {mse:.4f}")

    result = mlflow.register_model(
    f"runs:/{run.info.run_id}/model", "Deep NN"
)
print(f"Registered model version: {result.version}")
