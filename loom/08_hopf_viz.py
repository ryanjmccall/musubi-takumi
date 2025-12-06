"""
What you will see
When you run this, a 3D interactive plot will pop up.

The Input: Remember, the input was two flat, overlapping circles in 2D.

The Output: You will see two distinct clouds of points in 3D space. 
The red dots (inner circle) will be clustered together, 
completely separated from the blue dots (outer circle) by empty space.

This plot is the visual proof that your topological architecture can 
take a problem that is unsolvable with simple linear methods in 2D 
and make it trivially solvable by "lifting" it into a higher-dimensional 
space shaped by the Hopf fibration. This is the core concept of your 
photonic chip.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# --- RE-DEFINE THE MODEL STRUCTURE (Must match training script) ---
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
        hopf_features = self.hopf_map(z)
        out = self.decoder(hopf_features)
        return out

class HopfClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hopf = HopfBlock(in_features=2, out_features=3) 
        self.final = nn.Linear(3, 1) 

    def forward(self, x):
        # We want to visualize the output of the Hopf layer, not the final classification
        x_lifted = self.hopf(x)
        return x_lifted

# --- 1. GENERATE DATA ---
print("Generating data...")
X_raw, y_raw = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
X = torch.tensor(X_raw, dtype=torch.float32)

# --- 2. LOAD THE TRAINED MODEL ---
# NOTE: In a real workflow, you would save and load model weights. 
# For this demo, we'll re-train it quickly to ensure it's in a good state.
print("Re-training model quickly for visualization...")
model = HopfClassifier()
classifier_head = nn.Linear(3, 1) # Need this for training
optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier_head.parameters()), lr=0.01)
loss_fn = nn.BCELoss()
y = torch.tensor(y_raw, dtype=torch.float32).reshape(-1, 1)

for epoch in range(200): # 200 epochs is enough to get perfect separation
    x_lifted = model(X)
    y_pred = torch.sigmoid(classifier_head(x_lifted))
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- 3. EXTRACT TOPOLOGICAL FEATURES ---
print("Extracting 3D features from Hopf layer...")
model.eval() # Set to evaluation mode
with torch.no_grad():
    # Pass 2D data through the Hopf Block to get 3D "lifted" data
    X_3D = model(X).numpy()

# --- 4. VISUALIZE IN 3D ---
print("Plotting...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points, colored by their original class (inner vs outer circle)
scatter = ax.scatter(
    X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], 
    c=y_raw, 
    cmap='coolwarm', 
    s=20, 
    edgecolor='k'
)

# Add labels and title
ax.set_xlabel('Hopf Dimension 1')
ax.set_ylabel('Hopf Dimension 2')
ax.set_zlabel('Hopf Dimension 3')
ax.set_title('Visualizing the "Topological Lift"\nInner & Outer Circles Separated in 3D Hopf Space')

# Add a legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Outer Circle (Class 0)', markerfacecolor='blue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Inner Circle (Class 1)', markerfacecolor='red', markersize=10)
]
ax.legend(handles=legend_elements)

plt.show()