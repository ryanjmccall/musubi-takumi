import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.integrate import odeint

"""
MUSUBI-TAKUMI: MENTAL HEALTH TUNER
-------------------------------------------------------
A topological visualization of mental states in the frequency domain.
This script uses coupled oscillators and the Hopf map to model the 
stability of consciousness.

INSTRUCTIONS FOR USE (TOPOLOGICAL PLAY):
1. Run the script to launch the visualization dashboard.
2. Click "Depression": 
   - Observe the green line (consciousness topology) spiral inward and vanish. 
   - Visual Insight: "Snap out of it" fails because the system lacks the energy 
     to maintain a global loop.
3. Click "Hypomania": 
   - Observe the system explode into chaos, scrambling to cover the entire sphere.
   - Visual Insight: Unbounded energy creates a "fire" state with no coherent center.
4. The Magic Moment - Click "Therapy" (Radio Button):
   - No matter how chaotic or collapsed the state is, the math takes over.
   - Watch the line gently guide itself back into a stable, recurring loop (limit cycle).
   - Visual Definition of Healing: Not forcing the system to stop moving, 
     but finding a sustainable, resonant orbit.

STATES:
1. DEPRESSION (The High-Friction Well)
   - High Damping, Low Coupling.
   - System collapses inward; loss of global resonance.
   
2. HYPOMANIA (The Undamped Fire)
   - Negative Damping (Energy Gain), High Coupling.
   - System explodes outward; chaotic, unbounded energy.

3. HOMEOSTASIS (Therapy/Healing)
   - Dynamic non-linear damping.
   - Automatically regulates energy to maintain a stable "Window of Tolerance."
"""

# --- 1. System Dynamics ---
def system(state, t, m1, m2, k1, k2, k_couple, damping_val, mode):
    x1, v1, x2, v2 = state
    
    # Calculate Total Energy of the System (Kinetic + Potential)
    energy = 0.5 * (k1*x1**2 + m1*v1**2 + k2*x2**2 + m2*v2**2)
    
    # Determine Damping Coefficient based on Mode
    if mode == 'Manual':
        # User manually controls friction via slider
        c_dynamic = damping_val 
    elif mode == 'Therapy':
        # HOMEOSTATIC LOGIC:
        # Target Energy is the "Window of Tolerance" (e.g., Energy = 10.0)
        # If Energy < Target, friction becomes negative (pumping energy in).
        # If Energy > Target, friction becomes positive (dissipating excess).
        target_energy = 10.0
        sensitivity = 0.05
        c_dynamic = sensitivity * (energy - target_energy)

    # Equations of Motion (Coupled Harmonic Oscillators)
    dx1dt = v1
    dv1dt = (-k1*x1 - c_dynamic*v1 + k_couple*(x2 - x1)) / m1
    dx2dt = v2
    dv2dt = (-k2*x2 - c_dynamic*v2 - k_couple*(x2 - x1)) / m2
    
    return [dx1dt, dv1dt, dx2dt, dv2dt]

# --- 2. The Hopf Map (4D -> 3D Projection) ---
def hopf_map(x1, v1, x2, v2):
    z1 = x1 + 1j * v1
    z2 = x2 + 1j * v2
    
    # Avoid division by zero
    norm = np.sqrt(np.abs(z1)**2 + np.abs(z2)**2) + 1e-9
    z1 /= norm
    z2 /= norm
    
    X = 2 * (z1 * z2.conjugate()).real
    Y = 2 * (z1 * z2.conjugate()).imag
    Z = np.abs(z1)**2 - np.abs(z2)**2
    return X, Y, Z

# --- 3. Simulation Parameters ---
# Time steps
t_max = 50
dt = 0.05
t = np.arange(0, t_max, dt)

# Initial State
state0 = [2.0, 0.0, -1.0, 0.0] 

# Physics Constants
m1, m2 = 1.0, 1.0
k1, k2 = 10.0, 12.0 # Slightly detuned to create interesting beats

# Default Control Values
init_couple = 5.0
init_damping = 0.1
current_mode = 'Manual' # Start in manual mode

# --- 4. Visualization Setup ---
fig = plt.figure(figsize=(14, 7))
plt.subplots_adjust(bottom=0.25)
fig.suptitle("The Mental Health Tuner: Topological Stability", fontsize=16)

# Subplot 1: Phase Space (The Internal View)
ax_phase = fig.add_subplot(1, 2, 1)
ax_phase.set_title("Internal State (Phase Space)")
ax_phase.set_xlabel("Position")
ax_phase.set_ylabel("Velocity")
ax_phase.set_xlim(-4, 4)
ax_phase.set_ylim(-6, 6)
ax_phase.grid(True, alpha=0.3)
line1_phase, = ax_phase.plot([], [], 'b-', alpha=0.6, label='Self')
line2_phase, = ax_phase.plot([], [], 'r-', alpha=0.6, label='Other')
point1, = ax_phase.plot([], [], 'bo')
point2, = ax_phase.plot([], [], 'ro')
ax_phase.legend(loc='upper right')

# Subplot 2: Hopf Fibration (The Global/Holistic View)
ax_hopf = fig.add_subplot(1, 2, 2, projection='3d')
ax_hopf.set_title("Global Topology (Hopf Map)")
ax_hopf.set_xlim(-1, 1); ax_hopf.set_ylim(-1, 1); ax_hopf.set_zlim(-1, 1)
ax_hopf.view_init(elev=30, azim=45)

# Draw Reference Sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sphere = np.cos(u)*np.sin(v)
y_sphere = np.sin(u)*np.sin(v)
z_sphere = np.cos(v)
ax_hopf.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.1)

line_hopf, = ax_hopf.plot([], [], [], 'g-', linewidth=2, label='Consciousness')
point_hopf, = ax_hopf.plot([], [], [], 'go')

# --- 5. UI Controls ---
axcolor = 'lightgoldenrodyellow'

# Sliders
ax_couple = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
ax_damping = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)

s_couple = Slider(ax_couple, 'Connection (Coupling)', 0.0, 30.0, valinit=init_couple)
s_damping = Slider(ax_damping, 'Resistance (Friction)', -0.5, 1.0, valinit=init_damping)

# Buttons / Radio
ax_radio = plt.axes([0.02, 0.4, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(ax_radio, ('Manual', 'Therapy'), active=0)

def change_mode(label):
    global current_mode
    current_mode = label
radio.on_clicked(change_mode)

# Presets for ease of use
ax_depress = plt.axes([0.02, 0.25, 0.1, 0.04])
b_depress = Button(ax_depress, 'Depression', color='lightblue', hovercolor='0.975')

ax_mania = plt.axes([0.02, 0.19, 0.1, 0.04])
b_mania = Button(ax_mania, 'Hypomania', color='salmon', hovercolor='0.975')

def set_depression(event):
    s_couple.set_val(1.0)     # Low connection
    s_damping.set_val(0.8)    # High resistance
    radio.set_active(0)       # Switch to Manual

def set_mania(event):
    s_couple.set_val(25.0)    # Intense connection
    s_damping.set_val(-0.05)  # Negative resistance (energy gain)
    radio.set_active(0)       # Switch to Manual

b_depress.on_clicked(set_depression)
b_mania.on_clicked(set_mania)

# --- 6. Animation Logic ---
def update(frame):
    # We re-integrate every frame to allow real-time parameter changes.
    # In a full app, you might just integrate the next 'dt', but this is robust for small sims.
    
    # Get current slider values
    k_c = s_couple.val
    c_val = s_damping.val
    
    # Solve ODE
    sol = odeint(system, state0, t, args=(m1, m2, k1, k2, k_c, c_val, current_mode))
    
    x1 = sol[:, 0]
    v1 = sol[:, 1]
    x2 = sol[:, 2]
    v2 = sol[:, 3]
    
    # 2D Update
    line1_phase.set_data(x1, v1)
    line2_phase.set_data(x2, v2)
    point1.set_data([x1[-1]], [v1[-1]])
    point2.set_data([x2[-1]], [v2[-1]])
    
    # 3D Hopf Update
    X, Y, Z = hopf_map(x1, v1, x2, v2)
    line_hopf.set_data(X, Y)
    line_hopf.set_3d_properties(Z)
    point_hopf.set_data([X[-1]], [Y[-1]])
    point_hopf.set_3d_properties([Z[-1]])
    
    # Visual Feedback for Therapy Mode
    if current_mode == 'Therapy':
        ax_hopf.set_title("Global Topology: HOMEOSTATIC RECOVERY ACTIVE", color='green')
    else:
        ax_hopf.set_title("Global Topology: Manual Control", color='black')

    return line1_phase, line2_phase, line_hopf, point1, point2, point_hopf

ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=30, blit=False)
plt.show()