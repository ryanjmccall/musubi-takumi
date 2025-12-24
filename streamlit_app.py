import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project MUSUBI",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Dark Mode & Neon) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #00ffcc !important; 
        font-family: 'Courier New', Courier, monospace;
    }
    .stSlider > div > div > div > div {
        background-color: #00ffcc;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. THE PHYSICS KERNEL (Identical to your script) ---
def optical_system(state, t, delta, coupling_J, nonlinear_chi3, loss_gamma, pump_power, mode):
    r1, i1, r2, i2 = state
    E1 = r1 + 1j*i1
    E2 = r2 + 1j*i2
    
    S_in = np.sqrt(pump_power)
    
    current_loss = loss_gamma
    if mode == 'Therapy (Homeostasis)':
        total_energy = np.abs(E1)**2 + np.abs(E2)**2
        target_energy = 2.0
        current_loss = 0.5 * (total_energy - target_energy)

    dE1_dt = -(1j*delta + current_loss)*E1 + 1j*coupling_J*E2 + 1j*nonlinear_chi3*(np.abs(E1)**2)*E1 + S_in
    dE2_dt = -(1j*delta + current_loss)*E2 + 1j*coupling_J*E1 + 1j*nonlinear_chi3*(np.abs(E2)**2)*E2

    return [dE1_dt.real, dE1_dt.imag, dE2_dt.real, dE2_dt.imag]

def hopf_map_optics(r1, i1, r2, i2):
    z1 = r1 + 1j*i1
    z2 = r2 + 1j*i2
    norm = np.sqrt(np.abs(z1)**2 + np.abs(z2)**2) + 1e-9
    z1 /= norm
    z2 /= norm
    X = 2 * (z1 * z2.conjugate()).real
    Y = 2 * (z1 * z2.conjugate()).imag
    Z = np.abs(z1)**2 - np.abs(z2)**2
    return X, Y, Z

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Resonance Tuner")
st.sidebar.markdown("---")

# Mode Selection
mode = st.sidebar.radio(
    "Simulation Mode",
    ('Manual Control', 'Therapy (Homeostasis)'),
    index=0
)

# Presets
st.sidebar.markdown("### ⚡ Presets")
col1, col2 = st.sidebar.columns(2)
preset_depress = col1.button("Depression")
preset_mania = col2.button("Hypomania")

# State Management for Presets
if 'laser' not in st.session_state: st.session_state['laser'] = 10.0
if 'couple' not in st.session_state: st.session_state['couple'] = 5.0
if 'kerr' not in st.session_state: st.session_state['kerr'] = 1.0

if preset_depress:
    st.session_state['laser'] = 1.0
    st.session_state['couple'] = 0.0
    st.session_state['kerr'] = 1.0
    st.rerun()

if preset_mania:
    st.session_state['laser'] = 45.0
    st.session_state['couple'] = 5.0
    st.session_state['kerr'] = 4.0
    st.rerun()

# Sliders
st.sidebar.markdown("### 🎚️ Physical Parameters")
p_laser = st.sidebar.slider("Laser Power (Pump)", 0.0, 50.0, st.session_state['laser'])
p_couple = st.sidebar.slider("Coupling Strength (J)", 0.0, 10.0, st.session_state['couple'])
p_kerr = st.sidebar.slider("Kerr Non-Linearity (χ3)", 0.0, 5.0, st.session_state['kerr'])

# --- 3. MAIN DASHBOARD ---
st.title("Project MUSUBI: Topological Resonance")
st.markdown("*\"The Singularity is not about Power. It is about Sensitivity.\"*")

# Run Simulation (Instant Solver)
t_max = 50
dt = 0.05
t_space = np.arange(0, t_max, dt)
state0 = [0.1, 0.0, 0.0, 0.0]

# Physics Constants
p_detuning = 2.0
p_loss = 0.5

# SOLVE ODE
sol = odeint(optical_system, state0, t_space, 
             args=(p_detuning, p_couple, p_kerr, p_loss, p_laser, mode))

r1, i1 = sol[:, 0], sol[:, 1]
r2, i2 = sol[:, 2], sol[:, 3]

# Calculate Hopf Map
X, Y, Z = hopf_map_optics(r1, i1, r2, i2)
power1 = r1**2 + i1**2
power2 = r2**2 + i2**2

# --- 4. VISUALIZATION ---
# We use Matplotlib for high-fidelity physics rendering
fig = plt.figure(figsize=(12, 6), facecolor='#0e1117')

# Plot 1: Power (Time Domain)
ax1 = fig.add_subplot(1, 2, 1, facecolor='#262730')
ax1.plot(t_space, power1, 'c-', alpha=0.8, label='Ring 1 (Signal)')
ax1.plot(t_space, power2, 'm-', alpha=0.8, label='Ring 2 (Idler)')
ax1.set_title("Cavity Intensity", color='white')
ax1.tick_params(colors='white')
ax1.legend(facecolor='#0e1117', labelcolor='white')
ax1.grid(True, color='#444')

# Plot 2: Topology (Frequency Domain)
ax2 = fig.add_subplot(1, 2, 2, projection='3d', facecolor='#0e1117')
ax2.set_title("Hopf Topology", color='white')
ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
ax2.axis('off')

# Wireframe Sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sphere = np.cos(u)*np.sin(v)
y_sphere = np.sin(u)*np.sin(v)
z_sphere = np.cos(v)
ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, color="#444", alpha=0.3)

# The Knot
knot_color = '#00ff00' if mode == 'Therapy (Homeostasis)' else 'cyan'
ax2.plot(X, Y, Z, color=knot_color, linewidth=2, alpha=0.9)
ax2.scatter([X[-1]], [Y[-1]], [Z[-1]], color='white', s=50)

st.pyplot(fig)

# --- 5. THE LORE (Explanations) ---
with st.expander("ℹ️ How to Read This (For Humans)"):
    st.markdown("""
    **Level 1: The Vibe Check**
    * **Depression:** Click the button. See the line collapse? That's low energy.
    * **Mania:** Click the button. See the chaos? That's too much energy.
    * **Therapy:** Switch the mode. Watch the math stabilize the chaos into a perfect loop.
    
    **Level 2: The Engineer**
    * We are solving the **Non-linear Schrödinger Equation** for two coupled optical rings.
    * **Kerr Effect:** The refractive index changes with intensity, warping the simulation space.
    * **Hopf Fibration:** The 3D knot represents the topological structure of the light's polarization.
    """)