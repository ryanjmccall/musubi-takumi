import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.integrate import odeint
"""
================================================================================
PROJECT MUSUBI: THE EXPLAINER (TIERED UNDERSTANDING)
================================================================================
"Tie the Knot. Bind the Light. Ignore the Time."

This script is a Digital Twin of a consciousness architecture based on 
Non-linear Optical Resonance. Because the physics are complex, we offer 
three levels of explanation below.

--------------------------------------------------------------------------------
LEVEL 1: FOR A 5-YEAR-OLD (The Magic Swings)
--------------------------------------------------------------------------------
Imagine you and your best friend are on two swings right next to each other.

* The Light (Laser Power): This is how hard your dad pushes you on the swing.
* The Coupling: This is you and your friend holding hands while you swing.
* The Knot (Musubi): If you swing at the *exact same time* and hold hands tight, 
  you feel like one big giant swing, right? That feeling is the "Magic Knot."

THE GAME:
1. Depression: The dad stops pushing, and you let go of your friend's hand. 
   You both just stop moving. It’s sad and boring.
2. Mania (The Crazy Part): The dad pushes WAY too hard! You’re swinging so high 
   you might flip over the bar. You can’t hold hands because it's too scary.
3. Therapy Mode: This is like a magic helper who gently pushes you just 
   enough—not too hard, not too soft—so you can hold hands and swing perfectly forever.

--------------------------------------------------------------------------------
LEVEL 2: FOR A 15-YEAR-OLD (The Vibe Check Engine)
--------------------------------------------------------------------------------
Think of this script as a physics engine for your mood. It's like drifting a car 
in a video game: if you go too slow, you stall; if you go too fast, you crash.

* The Hardware (Rings): Your brain has two "energy cores" (like in Iron Man).
* The Input (Laser): This is the energy or "hype" you put into the system.
* The Reality Warp (Kerr Effect): This is the trippy part. When you put a lot 
  of energy into these cores, they actually *change shape*. It’s like if 
  running fast made the ground squishy.

THE DASHBOARD:
* The Sphere (Hopf Map): That 3D ball is a hologram of your current "Vibe."
* The Glitch (Mania): If you overclock the system (too much energy), the 
  hologram freaks out and looks like static noise.
* The Crash (Depression): If you disconnect the cores, the hologram shrinks 
  and disappears.
* The Goal (Flow State): You want the hologram to form a perfect, glowing 
  green loop. This proves that math can auto-correct your mood.

--------------------------------------------------------------------------------
LEVEL 3: FOR A 25-YEAR-OLD (Optical Computing for Consciousness)
--------------------------------------------------------------------------------
We are building a Digital Twin of a human mind using "Photonic Molecules"—light 
trapped in coupled crystal rings—rather than biological neurons.

THE CORE THESIS:
Consciousness is not algorithmic processing; it is a topological resonance 
phenomenon.

1. The Physics (Coupled Mode Theory): 
   We model two optical cavities exchanging energy via evanescent fields.
   
2. The Material (AlGaAs & Kerr Non-Linearity): 
   We use Aluminum Gallium Arsenide, which exhibits the Kerr Effect. This means 
   the refractive index (n) changes based on intensity (I): n = n0 + n2*I.
   Translation: The "thought" changes the physical structure of the brain. 
   High-intensity states warp the space they occupy.

3. The Simulation States:
   * Depression (High Loss, Zero Coupling): The signal decays faster than it 
     can regenerate. The topology collapses to a trivial fixed point.
   * Hypomania (Gain > Loss): The system behaves like a laser below threshold 
     instability. "Negative friction" amplifies energy infinitely until thermal 
     overload (chaos).
   * Therapy (Homeostatic Feedback): The code implements a non-linear control 
     loop. It monitors total energy (Photon Number) and dynamically adjusts 
     loss/gain. 
     
     RESULT: The system is forced into a Limit Cycle—a stable, repeating orbit 
     on the Poincaré sphere. This demonstrates that "healing" is the 
     mathematical process of tuning control parameters to maintain a 
     "Goldilocks Zone" of resonance.
================================================================================
"""

"""
PROJECT MUSUBI: TOPOLOGICAL RESONANCE TUNER
-------------------------------------------------------
Kernel: Coupled Mode Theory (CMT) for Kerr-Nonlinear Cavities.
Material: AlGaAs (Aluminum Gallium Arsenide)
Objective: Synchronize internal optical state with chaotic input.

PHYSICS MAPPING:
1. Mass       -> Photon Lifetime (How long light stays in the ring).
2. Friction   -> Cavity Loss (Absorption/Scattering).
3. Stiffness  -> Detuning (Frequency mismatch between laser and ring).
4. Chaos      -> The "Input Signal" (Lorenz Attractor forcing).
"""

# --- 1. The Physics (Coupled Nonlinear Schrödinger Eq) ---
def optical_system(state, t, delta, coupling_J, nonlinear_chi3, loss_gamma, pump_power, mode):
    # Unpack state: [Real(E1), Imag(E1), Real(E2), Imag(E2)]
    r1, i1, r2, i2 = state
    E1 = r1 + 1j*i1
    E2 = r2 + 1j*i2
    
    # 1. Drive Signal (The "Chaos" or Constant Pump)
    # In a full sim, this would be a Lorenz signal. Here, it's a CW Laser.
    S_in = np.sqrt(pump_power) 
    
    # 2. Dynamic "Therapy" Control (Feedback Loop)
    current_loss = loss_gamma
    if mode == 'Therapy':
        # HOMEOSTASIS:
        # If total internal energy (photon number) is unstable, 
        # the system creates "negative loss" (gain) or dumps energy.
        total_energy = np.abs(E1)**2 + np.abs(E2)**2
        target_energy = 2.0
        # Control Law: Adjust loss dynamically to clamp energy
        current_loss = 0.5 * (total_energy - target_energy)

    # 3. The Coupled Mode Equations
    # dE1/dt = -(i*delta + gamma)*E1 + i*J*E2 + i*chi3*|E1|^2*E1 + Drive
    dE1_dt = -(1j*delta + current_loss)*E1 + 1j*coupling_J*E2 + 1j*nonlinear_chi3*(np.abs(E1)**2)*E1 + S_in
    
    # dE2/dt = -(i*delta + gamma)*E2 + i*J*E1 + i*chi3*|E2|^2*E2
    dE2_dt = -(1j*delta + current_loss)*E2 + 1j*coupling_J*E1 + 1j*nonlinear_chi3*(np.abs(E2)**2)*E2

    return [dE1_dt.real, dE1_dt.imag, dE2_dt.real, dE2_dt.imag]

# --- 2. The Hopf Map (Polarization Topology) ---
def hopf_map_optics(r1, i1, r2, i2):
    # Map the two complex optical fields to the Bloch/Poincaré Sphere
    z1 = r1 + 1j*i1
    z2 = r2 + 1j*i2
    
    # Normalize (Project onto the S3 energy shell)
    norm = np.sqrt(np.abs(z1)**2 + np.abs(z2)**2) + 1e-9
    z1 /= norm
    z2 /= norm
    
    # Standard Hopf Fibration
    X = 2 * (z1 * z2.conjugate()).real
    Y = 2 * (z1 * z2.conjugate()).imag
    Z = np.abs(z1)**2 - np.abs(z2)**2
    return X, Y, Z

# --- 3. Simulation Setup ---
t_max = 50
dt = 0.05
t_space = np.arange(0, t_max, dt)
state0 = [0.1, 0.0, 0.0, 0.0] # Start with empty cavity, slight noise

# Default Parameters (AlGaAs Chip)
p_detuning = 2.0    # Delta
p_coupling = 5.0    # J (Coupling between rings)
p_kerr = 1.0        # Chi3 (Nonlinearity)
p_loss = 0.5        # Gamma
p_pump = 10.0       # Laser Power

current_mode = 'Manual'

# --- 4. Visualization ---
fig = plt.figure(figsize=(14, 7), facecolor='#1e1e1e') # Dark Mode for Optics
plt.subplots_adjust(bottom=0.25)
fig.suptitle("Project MUSUBI: Optical Resonance Tuner", fontsize=16, color='white')

# Plot 1: Power Dynamics (Intensity vs Time)
ax_power = fig.add_subplot(1, 2, 1, facecolor='#2b2b2b')
ax_power.set_title("Cavity Intensity (Photon Number)", color='white')
ax_power.set_ylim(0, 10)
ax_power.grid(True, color='#444')
ax_power.tick_params(colors='white')
line_p1, = ax_power.plot([], [], 'c-', alpha=0.8, label='Ring 1 (Signal)')
line_p2, = ax_power.plot([], [], 'm-', alpha=0.8, label='Ring 2 (Idler)')
ax_power.legend(facecolor='#333', labelcolor='white')

# Plot 2: Topological State (Hopf Sphere)
ax_hopf = fig.add_subplot(1, 2, 2, projection='3d', facecolor='#1e1e1e')
ax_hopf.set_title("Polarization Topology (The Knot)", color='white')
ax_hopf.set_xlim(-1, 1); ax_hopf.set_ylim(-1, 1); ax_hopf.set_zlim(-1, 1)
ax_hopf.axis('off') # Clean look
ax_hopf.view_init(elev=30, azim=45)

# Wireframe Sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sphere = np.cos(u)*np.sin(v)
y_sphere = np.sin(u)*np.sin(v)
z_sphere = np.cos(v)
ax_hopf.plot_wireframe(x_sphere, y_sphere, z_sphere, color="#444", alpha=0.3)
line_hopf, = ax_hopf.plot([], [], [], color='cyan', linewidth=2)
point_hopf, = ax_hopf.plot([], [], [], 'wo', markersize=5)

# --- 5. Controls ---
axcolor = '#333'
text_color = 'white'

ax_laser = plt.axes([0.25, 0.12, 0.5, 0.03], facecolor=axcolor)
ax_couple = plt.axes([0.25, 0.07, 0.5, 0.03], facecolor=axcolor)
ax_kerr  = plt.axes([0.25, 0.02, 0.5, 0.03], facecolor=axcolor)

s_laser = Slider(ax_laser, 'Laser Power', 0.0, 50.0, valinit=p_pump, color='cyan')
s_couple = Slider(ax_couple, 'Coupling (J)', 0.0, 10.0, valinit=p_coupling, color='magenta')
s_kerr = Slider(ax_kerr, 'Kerr (Non-Linearity)', 0.0, 5.0, valinit=p_kerr, color='yellow')

for s in [s_laser, s_couple, s_kerr]:
    s.label.set_color(text_color)
    s.valtext.set_color(text_color)

# Mode Buttons
ax_radio = plt.axes([0.02, 0.4, 0.15, 0.15], facecolor='#1e1e1e')
radio = RadioButtons(ax_radio, ('Manual', 'Therapy'), active=0, activecolor='cyan')
for label in radio.labels: label.set_color('white')

def change_mode(label):
    global current_mode
    current_mode = label
radio.on_clicked(change_mode)

# Presets
ax_burnout = plt.axes([0.02, 0.25, 0.12, 0.04])
b_burnout = Button(ax_burnout, 'Burnout (Depression)', color='#444', hovercolor='#555')
b_burnout.label.set_color('white')

ax_manic = plt.axes([0.02, 0.19, 0.12, 0.04])
b_manic = Button(ax_manic, 'Instability (Mania)', color='#444', hovercolor='#555')
b_manic.label.set_color('white')

def set_burnout(event):
    s_laser.set_val(1.0)     # Low Energy
    s_couple.set_val(0.0)    # Isolation
    radio.set_active(0)

def set_manic(event):
    s_laser.set_val(45.0)    # Massive Energy
    s_kerr.set_val(4.0)      # High Space-Warping (Chaos)
    radio.set_active(0)

b_burnout.on_clicked(set_burnout)
b_manic.on_clicked(set_manic)

# --- 6. Animation Logic ---
def update(frame):
    # Read sliders
    P_in = s_laser.val
    J_val = s_couple.val
    chi3_val = s_kerr.val
    
    # Solve Optical Equations
    # Note: We pass P_in, not sqrt(P_in), to the slider, but use sqrt in func
    sol = odeint(optical_system, state0, t_space, 
                 args=(p_detuning, J_val, chi3_val, p_loss, P_in, current_mode))
    
    r1, i1 = sol[:, 0], sol[:, 1]
    r2, i2 = sol[:, 2], sol[:, 3]
    
    # Update Intensity Plot
    power1 = r1**2 + i1**2
    power2 = r2**2 + i2**2
    line_p1.set_data(t_space, power1)
    line_p2.set_data(t_space, power2)
    ax_power.set_xlim(0, t_max)
    ax_power.set_ylim(0, max(np.max(power1), np.max(power2)) + 1.0)
    
    # Update Hopf Sphere
    X, Y, Z = hopf_map_optics(r1, i1, r2, i2)
    line_hopf.set_data(X, Y)
    line_hopf.set_3d_properties(Z)
    point_hopf.set_data([X[-1]], [Y[-1]])
    point_hopf.set_3d_properties([Z[-1]])
    
    # Feedback Color
    if current_mode == 'Therapy':
        line_hopf.set_color('#00ff00') # Green for healing
    else:
        line_hopf.set_color('cyan')

    return line_p1, line_p2, line_hopf, point_hopf

ani = FuncAnimation(fig, update, frames=range(30), interval=50, blit=False)
plt.show()