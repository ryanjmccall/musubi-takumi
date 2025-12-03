import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def hopf_inverse(theta, phi, alpha_steps=100):
    """
    Maps a point (theta, phi) on S2 to a circle (fiber) in S3,
    then projects S3 -> R3 via stereographic projection.
    """
    # Array of angles for the fiber (the "time" or circle parameter)
    alpha = np.linspace(0, 2 * np.pi, alpha_steps)
    
    # 1. Map S2 (theta, phi) to complex coords (z0, z1) in S3
    # We use a half-angle parameterization for the base point
    eta = theta / 2.0
    
    # The fiber is e^(i*alpha) * (z0, z1)
    # z0 = cos(eta) * e^(i * (phi + alpha))
    # z1 = sin(eta) * e^(i * alpha)
    
    # 2. Compute the 4D coordinates (x1, x2, x3, x4)
    # z0 = x1 + i*x2, z1 = x3 + i*x4
    x1 = np.cos(eta) * np.cos(phi + alpha)
    x2 = np.cos(eta) * np.sin(phi + alpha)
    x3 = np.sin(eta) * np.cos(alpha)
    x4 = np.sin(eta) * np.sin(alpha)
    
    # 3. Stereographic projection from S3 to R3
    # factor = 1 / (1 - x4) usually, or similar projection
    denom = 1 - x4
    
    # Handle division by zero for the pole
    with np.errstate(divide='ignore', invalid='ignore'):
        X = x1 / denom
        Y = x2 / denom
        Z = x3 / denom
        
    return X, Y, Z

def plot_hopf_fibration(n_fibers=20):
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.axis('off')

    # Generate fibers by sampling theta (0 to pi) and phi (0 to 2pi)
    thetas = np.linspace(0.1, np.pi-0.1, n_fibers) # avoid poles
    
    # Use a colormap for artistic effect
    colors = plt.cm.hsv(np.linspace(0, 1, n_fibers))

    for i, theta in enumerate(thetas):
        # We rotate phi slightly for each fiber to create a spiral effect
        phi = i * (np.pi / 4) 
        X, Y, Z = hopf_inverse(theta, phi)
        ax.plot(X, Y, Z, color=colors[i], lw=1.5, alpha=0.8)

    # Set limits to keep aspect ratio decent
    limit = 3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    plt.show()

if __name__ == "__main__":
    plot_hopf_fibration(n_fibers=50)