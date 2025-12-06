import torch
import torch.nn as nn

class HopfBlock(nn.Module):
    """
    A Neural Network Layer that simulates the topology of a Hopf Fibration.
    
    Concept:
    It takes an input vector, projects it into a 4D 'hypersphere' space (C^2),
    applies the Hopf Map to project it down to a 3D 'spherical' space (R^3),
    and then projects it to the desired output dimension.
    
    This acts as a 'topological non-linearity' replacing standard ReLUs.
    """
    def __init__(self, in_features, out_features):
        super(HopfBlock, self).__init__()
        
        # 1. ENCODER: Map input data to the 4D "Hopf Space" 
        # (This represents modulating the electronic signal into light/complex amplitudes)
        # We target 4 dims because the Hopf fibration maps S3 (in R4) -> S2 (in R3)
        self.encoder = nn.Linear(in_features, 4)
        
        # 2. DECODER: Map the 3D result of the Hopf function back to the user's output dim
        self.decoder = nn.Linear(3, out_features)

    def hopf_map(self, z):
        """
        The mathematical heart.
        Input z: shape (batch, 4) -> representing 2 complex numbers z0, z1
        
        z = (x1, y1, x2, y2) representing z0 = x1 + iy1, z1 = x2 + iy2
        
        The Hopf Map h: S3 -> S2 is defined as:
        h(z0, z1) = (2 * z0 * conj(z1), |z0|^2 - |z1|^2)
        
        Decomposed into real coordinates (X, Y, Z):
        X = 2(x1*x2 + y1*y2)
        Y = 2(x1*y2 - x2*y1)
        Z = (x1^2 + y1^2) - (x2^2 + y2^2)
        """
        x1 = z[:, 0]
        y1 = z[:, 1]
        x2 = z[:, 2]
        y2 = z[:, 3]

        # The topological mixing logic
        X_out = 2 * (x1 * x2 + y1 * y2)
        Y_out = 2 * (x1 * y2 - x2 * y1)
        Z_out = (x1**2 + y1**2) - (x2**2 + y2**2)

        # Stack them into a (batch, 3) tensor
        return torch.stack((X_out, Y_out, Z_out), dim=1)

    def forward(self, x):
        # Step 1: Project to 4D "Hopf Space"
        z = self.encoder(x)
        
        # Step 2: Apply the Topological Logic (The "Chip" Simulation)
        hopf_features = self.hopf_map(z)
        
        # Step 3: Readout to desired output size
        out = self.decoder(hopf_features)
        
        return out

# --- DEMO: Run a quick test ---
if __name__ == "__main__":
    # Create the simulator layer
    # Let's say we have 10 input features and want 5 output features
    mock_chip = HopfBlock(in_features=10, out_features=5)

    # Create fake data (batch of 2 samples, 10 features each)
    input_data = torch.randn(2, 10)

    # Run the simulator
    output = mock_chip(input_data)

    print("Input shape:", input_data.shape)
    print("Output shape:", output.shape)
    print("\nSuccess. The data flowed through the simulated Hopf fibration.")
    print("Output values:\n", output)