# Hybrid Approach for Anomaly Detection using Autoencoder + NPLM in PyTorch.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Required Libraries
# Make sure to install the following libraries:
# pip install torch torchvision pandas matplotlib numpy plotly
# File Paths
data_path_train = "O:/AI projects/NPLM/train.csv"
data_path_test = "O:/AI projects/NPLM/test.csv"
# Step 1: Load and Preprocess the Dataset
def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Combine train and test for anomaly detection
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Extract relevant columns and aggregate duplicates by taking mean
    data = combined_df.groupby(['store', 'item'])['sales'].mean().unstack(fill_value=0)
    
    # Normalize data
    data = (data - data.mean()) / data.std()
    
    return torch.FloatTensor(data.values)

# Load the dataset
data = load_and_preprocess_data(data_path_train, data_path_test)
# Step 2: Define the Autoencoder Model (Your Model)
class Autoencoder(nn.Module):
    def __init__(self, input_dim=50):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def summary(self, input_size):
        from torchsummary import summary
        summary(self, input_size)
        
# Step 3: Define the NPLM Model (My Model)
class NPLMNet(nn.Module):
    def __init__(self, input_dim=50):
        super(NPLMNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

    def summary(self, input_size):
        from torchsummary import summary
        summary(self, input_size)
# Step 4: Train the Autoencoder
autoencoder = Autoencoder(input_dim=data.size(1))
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    autoencoder.train()
    optimizer.zero_grad()
    output = autoencoder(data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# Step 5: Use Autoencoder for Reconstruction and Filter Obvious Anomalies
with torch.no_grad():
    autoencoder.eval()
    reconstructed_data = autoencoder(data)
    mse = torch.mean((data - reconstructed_data) ** 2, dim=1)

# Set threshold based on MSE (e.g., 95th percentile)
mse_threshold = torch.quantile(mse, 0.95)
filtered_data = data[mse < mse_threshold]  # Keep only less obvious anomalies
print(f"Filtered data size after Autoencoder: {filtered_data.size(0)}")
# Step 6: Train NPLM on Filtered Data
reference_data = data  # Reference dataset
nplm = NPLMNet(input_dim=data.size(1))
nplm_optimizer = optim.Adam(nplm.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    nplm.train()
    nplm_optimizer.zero_grad()
    loss = torch.sum(torch.exp(nplm(reference_data)) - 1) - torch.sum(nplm(filtered_data))
    loss.backward()
    nplm_optimizer.step()
    print(f"NPLM Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# Step 7: Use NPLM to Score Anomalies
with torch.no_grad():
    nplm.eval()
    anomaly_scores = nplm(data).squeeze()

# Set a threshold for anomaly detection (e.g., 95th percentile of scores)
score_threshold = torch.quantile(anomaly_scores, 0.95)
detected_anomalies = data[anomaly_scores > score_threshold]
# Step 8: Interactive Visualization using Plotly
# Histogram of Anomaly Scores
fig = px.histogram(
    x=anomaly_scores.numpy(),
    nbins=50,
    title="NPLM Anomaly Scores",
    labels={"x": "Score", "y": "Frequency"}
)
fig.add_vline(x=score_threshold.item(), line_dash="dash", line_color="red", annotation_text="Threshold")
fig.show()

# Scatter plot of detected anomalies
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=anomaly_scores.numpy(),
    mode='markers',
    marker=dict(color=['red' if score > score_threshold else 'blue' for score in anomaly_scores.numpy()]),
    name='Anomaly Scores'
))
fig.update_layout(
    title="Scatter Plot of Anomaly Scores",
    xaxis_title="Sample Index",
    yaxis_title="Anomaly Score"
)
fig.show()

# Output total anomalies detected
print(f"Total detected anomalies: {detected_anomalies.size(0)}")

# Display Model Summaries
print("\nAutoencoder Model Summary:")
autoencoder.summary((1, data.size(1)))

print("\nNPLM Model Summary:")
nplm.summary((1, data.size(1)))
