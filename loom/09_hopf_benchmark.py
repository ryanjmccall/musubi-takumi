import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.datasets import make_circles

# --- 1. DEFINE THE CONTENDERS ---

# CONTENDER A: The Hopf Architecture (~28 Parameters)
class HopfBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(HopfBlock, self).__init__()
        self.encoder = nn.Linear(in_features, 4) # 2->4
        self.decoder = nn.Linear(3, out_features) # 3->3

    def hopf_map(self, z):
        x1, y1, x2, y2 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        X_out = 2 * (x1 * x2 + y1 * y2)
        Y_out = 2 * (x1 * y2 - x2 * y1)
        Z_out = (x1**2 + y1**2) - (x2**2 + y2**2)
        return torch.stack((X_out, Y_out, Z_out), dim=1)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(self.hopf_map(z))

# CONTENDER A: The "Widened" Hopf Architecture (~43 Parameters)
class HopfModel(nn.Module):
    def __init__(self):
        super().__init__()
        # We changed out_features from 3 -> 6 to give it more "readout" power
        self.hopf = HopfBlock(in_features=2, out_features=6)
        self.final = nn.Linear(6, 1) # Reads the 6 features -> 1 score
    
    def forward(self, x):
        return torch.sigmoid(self.final(self.hopf(x)))

# CONTENDER B: The "Starved" MLP (Fair Fight)
# We reduce Hidden Layer to 10 neurons to match Hopf's size.
# Params: (2*10+10) + (10*1+1) = 30 + 11 = 41 Parameters.
# This makes it a fair fight: 43 Params (Hopf) vs 41 Params (MLP).
class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),  # Reduced from 16 to 10
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# --- 2. UTILS ---

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, X, y, epochs=1000, name="Model"):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    start_time = time.time()
    
    for i in range(epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Early stopping check for efficiency metric
        acc = ((y_pred > 0.5) == y).float().mean()
        if acc >= 0.99:
            print(f"   -> {name} solved it at Epoch {i}!")
            return i, acc.item()
            
    return epochs, acc.item()

# --- 3. THE SHOWDOWN ---

# Setup Data
X_raw, y_raw = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
X = torch.tensor(X_raw, dtype=torch.float32)
y = torch.tensor(y_raw, dtype=torch.float32).reshape(-1, 1)

# Initialize
hopf_model = HopfModel()
mlp_model = StandardMLP()

hopf_params = count_parameters(hopf_model)
mlp_params = count_parameters(mlp_model)

print(f"--- BENCHMARK START ---")
print(f"Hopf Parameters: {hopf_params}")
print(f"MLP Parameters:  {mlp_params}")
print(f"Ratio: Hopf is using {hopf_params/mlp_params:.2f}x the parameters of the MLP.")
print("-" * 30)

# Run Hopf
print("Training Hopf Model...")
hopf_epochs, hopf_acc = train_model(hopf_model, X, y, name="Hopf")

# Run MLP
print("Training Standard MLP...")
mlp_epochs, mlp_acc = train_model(mlp_model, X, y, name="MLP")

# --- 4. RESULTS ---
print("\n" + "="*30)
print("FINAL SCORECARD")
print("="*30)
print(f"Hopf Model | Params: {hopf_params} | Converged: Epoch {hopf_epochs} | Acc: {hopf_acc*100:.1f}%")
print(f"Std MLP    | Params: {mlp_params} | Converged: Epoch {mlp_epochs} | Acc: {mlp_acc*100:.1f}%")

if hopf_params < mlp_params and hopf_epochs <= mlp_epochs:
    print("\nWINNER: Hopf Architecture.")
    print(f"Reason: Solved the problem faster using {100 - (hopf_params/mlp_params)*100:.1f}% fewer parameters.")