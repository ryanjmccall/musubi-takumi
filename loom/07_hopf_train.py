import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles

# --- PART 1: THE SIMULATOR (Formerly Script 6) ---

class HopfBlock(nn.Module):
    """
    A Neural Network Layer that simulates the topology of a Hopf Fibration.
    Input -> Project to 4D (Complex Space) -> Hopf Map (S3 to S2) -> Output
    """
    def __init__(self, in_features, out_features):
        super(HopfBlock, self).__init__()
        
        # 1. ENCODER: Map input data to the 4D "Hopf Space" 
        self.encoder = nn.Linear(in_features, 4)
        
        # 2. DECODER: Map the 3D result of the Hopf function back to output dim
        self.decoder = nn.Linear(3, out_features)

    def hopf_map(self, z):
        """
        The mathematical heart.
        Input z: shape (batch, 4) -> representing 2 complex numbers z0, z1
        h(z0, z1) = (2z0z1*, |z0|^2 - |z1|^2)
        """
        x1, y1, x2, y2 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]

        # The topological mixing logic (Hopf Map equations)
        X_out = 2 * (x1 * x2 + y1 * y2)
        Y_out = 2 * (x1 * y2 - x2 * y1)
        Z_out = (x1**2 + y1**2) - (x2**2 + y2**2)

        return torch.stack((X_out, Y_out, Z_out), dim=1)

    def forward(self, x):
        z = self.encoder(x)
        hopf_features = self.hopf_map(z)
        out = self.decoder(hopf_features)
        return out

# --- PART 2: THE SMOKE TEST (Formerly Script 7) ---

# 1. PREPARE DATA: Two concentric circles (The "XOR" Problem)
print("Generating data...")
X_raw, y_raw = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
X = torch.tensor(X_raw, dtype=torch.float32)
y = torch.tensor(y_raw, dtype=torch.float32).reshape(-1, 1)

# 2. DEFINE THE MODEL
class HopfClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 2 coordinates (x, y)
        # Hidden: HopfBlock projects to 4D -> mixes -> outputs 3D
        self.hopf = HopfBlock(in_features=2, out_features=3) 
        # Output: 1 score (0 or 1)
        self.final = nn.Linear(3, 1) 

    def forward(self, x):
        x = self.hopf(x)
        return torch.sigmoid(self.final(x))

model = HopfClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# 3. TRAINING LOOP
print("Training Hopf Classifier...")
history = []

for epoch in range(1001):
    # Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        acc = ((y_pred > 0.5) == y).float().mean()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc:.2f}")

# 4. FINAL VERIFICATION
final_acc = ((model(X) > 0.5) == y).float().mean().item()
print(f"\nFinal Accuracy: {final_acc * 100:.2f}%")

if final_acc > 0.95:
    print("SUCCESS: The Hopf Architecture is learning non-linear topology.")
else:
    print("WARNING: Convergence issues detected.")