import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# --- 1. DEFINE THE CONTENDERS (Same "Fair Fight" config) ---

class HopfBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(HopfBlock, self).__init__()
        self.encoder = nn.Linear(in_features, 4)
        self.decoder = nn.Linear(3, out_features)

    def hopf_map(self, z):
        x1, y1, x2, y2 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        X_out = 2 * (x1 * x2 + y1 * y2)
        Y_out = 2 * (x1 * y2 - x2 * y1)
        Z_out = (x1**2 + y1**2) - (x2**2 + y2**2)
        return torch.stack((X_out, Y_out, Z_out), dim=1)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(self.hopf_map(z))

class HopfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hopf = HopfBlock(in_features=2, out_features=6) # 43 Params
        self.final = nn.Linear(6, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.final(self.hopf(x)))

class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 41 Params (The "Fair" Config)
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# --- 2. TRAIN HELPERS ---
def train_until_converged(model, X, y, max_epochs=500):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    for i in range(max_epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ((y_pred > 0.5) == y).float().mean() >= 0.99:
            return model
    return model

# --- 3. EXECUTION ---
# Data
X_raw, y_raw = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
X = torch.tensor(X_raw, dtype=torch.float32)
y = torch.tensor(y_raw, dtype=torch.float32).reshape(-1, 1)

# Train
print("Training models for visualization...")
hopf = train_until_converged(HopfModel(), X, y)
mlp = train_until_converged(StandardMLP(), X, y)

# --- 4. VISUALIZATION OF BOUNDARIES ---
def plot_boundary(model, ax, title):
    # Create a grid covering the space
    h = 0.02
    x_min, x_max = X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5
    y_min, y_max = X_raw[:, 1].min() - 0.5, X_raw[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on the grid
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid_tensor).reshape(xx.shape)
        
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw, cmap=plt.cm.RdBu_r, edgecolors='k', s=20)
    ax.set_title(title)

print("Plotting decision boundaries...")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

plot_boundary(hopf, axs[0], "Hopf Architecture (Smooth Topology)")
plot_boundary(mlp, axs[1], "Standard MLP (Linear Approximation)")

plt.show()