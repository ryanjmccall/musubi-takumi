import numpy as np
import plotly.graph_objects as go
from scipy.special import genlaguerre

# --- KNOBS: PLAY WITH THESE ---
# The physics is defined by the interference of two slightly detuned colors.
LAMBDA_1 = 1.0          # Base wavelength (microns)
LAMBDA_RATIO = 100/101  # Ratio lambda2/lambda1 (Controls the 'Time' beat period)
WAIST_W0 = 10.0         # Beam waist radius (microns)
GRID_RES = 40           # Resolution (Lower = Faster, Higher = Smoother)
TIME_STEPS = 40         # How many time slices to simulate
P_ORDER = 1             # Topological charge P (Spatial complexity)

# --- DARK AUTUMN PALETTE ---
COLORS = ['#542516', '#7D5832', '#9E451C', '#3C4A28', '#204443', '#A67B34']

def lg_mode(r, phi, p, l, w0, z=0, k=1):
    """
    Calculates the Laguerre-Gaussian mode amplitude at z=0.
    """
    # Normalized radial coordinate
    rho = (r * np.sqrt(2)) / w0
    
    # Laguerre polynomial L_p^|l|
    laguerre = genlaguerre(p, abs(l))(2 * r**2 / w0**2)
    
    # Envelope and Phase
    amplitude = (rho ** abs(l)) * laguerre * np.exp(-r**2 / w0**2)
    phase = np.exp(1j * l * phi)
    
    return amplitude * phase

def calculate_field(x, y, t, l1, l2):
    """
    Implements Eq (2) from the paper: Superposition of bichromatic modes.
    Source: Physical Review Letters 135, 083801 (2025)
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Frequencies (c=1 for sim units)
    omega1 = 2 * np.pi / l1
    omega2 = 2 * np.pi / l2
    
    # --- The Topological Recipe (Eq 2) ---
    # Right Circular (R): LG00(w1) + LG10(w2)
    # The 'p=1' mode in LG10 provides the radial node needed for nesting
    E_R = (lg_mode(r, phi, p=0, l=0, w0=WAIST_W0) * np.exp(-1j * omega1 * t) + 
           lg_mode(r, phi, p=1, l=0, w0=WAIST_W0) * np.exp(-1j * omega2 * t))

    # Left Circular (L): LG01(w1) - LG01(w2)
    # The 'l=1' mode provides the OAM vortex
    E_L = (lg_mode(r, phi, p=0, l=P_ORDER, w0=WAIST_W0) * np.exp(-1j * omega1 * t) - 
           lg_mode(r, phi, p=0, l=P_ORDER, w0=WAIST_W0) * np.exp(-1j * omega2 * t))
           
    return E_R, E_L

def get_stokes_azimuth(E_R, E_L):
    """
    Calculates the azimuth of the pseudospin vector on the Poincare sphere.
    """
    # Stokes parameters from Circular Basis
    # s1 = 2 Re(E_R * E_L*)
    # s2 = 2 Im(E_R * E_L*)
    # Azimuth = arctan2(s2, s1)
    
    term = E_R * np.conj(E_L)
    s1 = 2 * np.real(term)
    s2 = 2 * np.imag(term)
    
    return np.arctan2(s2, s1)

def generate_spacetime_hopfion():
    # 1. Setup Space-Time Grid
    limit = 25
    x = np.linspace(-limit, limit, GRID_RES)
    y = np.linspace(-limit, limit, GRID_RES)
    
    # Time limits determined by beat period T = lambda1 * lambda2 / |l1-l2|
    # With ratio 100/101, period is large. We verify one "beat".
    beat_period = 100 
    t = np.linspace(-beat_period/2, beat_period/2, TIME_STEPS)
    
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # 2. Compute Fields
    lambda2 = LAMBDA_1 * LAMBDA_RATIO
    E_R, E_L = calculate_field(X, Y, T, LAMBDA_1, lambda2)
    
    # 3. Compute Spin Texture (Azimuth maps to color/fiber)
    azimuth = get_stokes_azimuth(E_R, E_L)
    
    # 4. Filter for specific "fibers" (isosurfaces of constant azimuth)
    # This visualizes the knot structure
    fig = go.Figure()
    
    target_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    fiber_names = ["Fiber 0", "Fiber 90", "Fiber 180", "Fiber 270"]
    
    for i, angle in enumerate(target_angles):
        # Create a mask for points close to this angle
        # We use isosurface for smooth visualization
        
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=T.flatten(),
            value=np.cos(azimuth.flatten() - angle), # Maximize when angle matches
            isomin=0.95,
            isomax=1.0,
            surface_count=2,
            colorscale=[[0, COLORS[i % len(COLORS)]], [1, COLORS[i % len(COLORS)]]],
            showscale=False,
            opacity=0.6,
            name=fiber_names[i]
        ))

    # Layout
    fig.update_layout(
        title="Space-Time Optical Hopfion (Eq. 2 Visualization)",
        scene=dict(
            xaxis_title='X (Space)',
            yaxis_title='Y (Space)',
            zaxis_title='cT (Time)',
            bgcolor='black'
        ),
        template='plotly_dark',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.show()

if __name__ == "__main__":
    generate_spacetime_hopfion()