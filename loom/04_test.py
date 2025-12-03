"""
Project MUSUBI: Topological Resonance & Singularity Simulator
=============================================================

"We just need to Vibe in the frequency domain."

This script serves as a 'Digital Twin' for a proposed nonlinear nanophotonic 
experiment. It simulates the synchronization of a passive optical microcavity 
(AlGaAs Ring Resonator) with a chaotic driver (Lorenz Attractor).

HARD MODE ENABLED:
------------------
This version simulates physical limitations (Thermal Loading, Mode Mismatch).
Maxing out the sliders will destabilize or 'melt' the chip. You must find
the 'Goldilocks Zone' to achieve Singularity.

Usage
-----
Run the script to launch the interactive "Resonance Tuner" UI.
Adjust the sliders to maximize the Correlation Score (> 90%).

Dependencies: numpy, scipy, matplotlib
Author: Musubi Takumi / [Your Name]
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider
import warnings

# Suppress ODE integration warnings for "Game Feel" (simulating instability)
warnings.filterwarnings("ignore")

# --- 1. THE PHYSICS ENGINE (HARD MODE) ---
def system_dynamics_6D(state, t, params):
    # Unpack state
    x, y, z, ar_H, ai_H, ar_V, ai_V = state
    a_H = ar_H + 1j * ai_H
    a_V = ar_V + 1j * ai_V
    
    # Unpack Parameters
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    
    # HARD MODE PHYSICS:
    # 1. Thermal Loading: High Pump energy heats the chip, ruining the Q-factor.
    #    As Pump increases, 'kappa' (loss) increases.
    base_kappa = 1.0
    pump_heat_penalty = 0.05 * params['pump'] 
    effective_kappa = base_kappa + pump_heat_penalty
    
    chi3 = params['chi3']
    gamma = params['gamma']
    pump = params['pump']
    coupling_J = 5.0 

    # --- Island A: Lorenz Chaos ---
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    # --- The Bridge: Coupling ---
    # The Chaos X modifies the refractive index (Detuning)
    detuning_H = gamma * x
    detuning_V = gamma * (z - rho)

    # --- Island B: Optical Cavity (Coupled Mode Theory) ---
    # Note: We use 'effective_kappa' to simulate thermal degradation
    
    # Horizontal Mode
    daH_dt = (1j * detuning_H - effective_kappa/2) * a_H \
             - 1j * chi3 * (np.abs(a_H)**2 + 2*np.abs(a_V)**2) * a_H \
             + 1j * coupling_J * a_V + pump

    # Vertical Mode
    daV_dt = (1j * detuning_V - effective_kappa/2) * a_V \
             - 1j * chi3 * (np.abs(a_V)**2 + 2*np.abs(a_H)**2) * a_V \
             + 1j * coupling_J * a_H 

    return [dx_dt, dy_dt, dz_dt, daH_dt.real, daH_dt.imag, daV_dt.real, daV_dt.imag]

def run_simulation(pump, gamma, chi3):
    params = {'pump': pump, 'gamma': gamma, 'chi3': chi3}
    # Keep simulation short (30 time units) for UI responsiveness
    t = np.linspace(0, 30, 2000) 
    state0 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        sol = odeint(system_dynamics_6D, state0, t, args=(params,))
        return sol
    except:
        # If solver crashes (instability), return zeros
        return np.zeros((len(t), 7))

# --- 2. THE GAME UI SETUP ---
# Use a dark theme for that "Dark Autumn" Sci-Fi aesthetic
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 8), facecolor='#121212')
plt.subplots_adjust(bottom=0.25) # Space for sliders

# Plot A: Synchronization (Input vs Output)
ax_sync = fig.add_subplot(121, facecolor='#1e1e1e')
ax_sync.set_title("ISLAND A vs ISLAND B (Resonance)", color='white', fontsize=12)
ax_sync.set_xlabel("Input Chaos (X)", color='gray')
ax_sync.set_ylabel("Output Intensity (I)", color='gray')
ax_sync.grid(True, color='#333333', linestyle='--', alpha=0.5)
line_sync, = ax_sync.plot([], [], '.', color='#ff0055', ms=1.5, alpha=0.6)

# Plot B: Poincaré Sphere (Topological State)
ax_sphere = fig.add_subplot(122, projection='3d', facecolor='#121212')
ax_sphere.set_axis_off() # Hide axes for clean look

# Draw static wireframe sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_s = np.cos(u)*np.sin(v)
y_s = np.sin(u)*np.sin(v)
z_s = np.cos(v)
ax_sphere.plot_wireframe(x_s, y_s, z_s, color="#444444", alpha=0.2)
line_output, = ax_sphere.plot([], [], [], color='#00ffcc', lw=1.5, alpha=0.9)

# Text Labels
title_text = ax_sphere.text2D(0.5, 0.95, "STATUS: INITIALIZING...", 
                              transform=ax_sphere.transAxes, color='white', 
                              ha='center', fontsize=14, weight='bold')

score_text = fig.text(0.5, 0.88, "RESONANCE: 0.0%", ha='center', 
                      fontsize=22, color='gray', weight='bold')

# --- 3. SLIDERS & CONTROLS ---
axcolor = '#333333'
ax_pump = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
ax_gamma = plt.axes([0.25, 0.06, 0.5, 0.03], facecolor=axcolor)
ax_chi = plt.axes([0.25, 0.02, 0.5, 0.03], facecolor=axcolor)

# Ranges set to allow "breaking" the system (max is dangerous)
s_pump = Slider(ax_pump, 'Laser Power (P)', 0.1, 100.0, valinit=20.0, color='#00ffcc')
s_gamma = Slider(ax_gamma, 'Coupling (y)', 0.0, 5.0, valinit=1.5, color='#ff0055')
s_chi = Slider(ax_chi, 'Non-Linearity (X3)', 0.0, 2.0, valinit=0.5, color='#ffff00')

# Style the slider text
for s in [s_pump, s_gamma, s_chi]:
    s.label.set_color('white')
    s.valtext.set_color('white')

# --- 4. UPDATE LOGIC (THE GOLDILOCKS ALGORITHM) ---
def update(val):
    # Get knob values
    P = s_pump.val
    G = s_gamma.val
    C = s_chi.val
    
    # Run the physics
    sol = run_simulation(P, G, C)
    
    # Safety Check: Did the solver return NaNs (Explosion)?
    if np.isnan(sol).any():
        title_text.set_text("STATUS: CRITICAL ERROR (NaN)")
        score_text.set_text("RESONANCE: ERROR")
        score_text.set_color('red')
        return

    # Extract Data
    x_in = sol[:, 0]
    
    # Stokes Parameters Calculation
    ar_H, ai_H, ar_V, ai_V = sol[:, 3], sol[:, 4], sol[:, 5], sol[:, 6]
    a_H = ar_H + 1j * ai_H
    a_V = ar_V + 1j * ai_V
    
    # Intensity S0
    S0 = np.abs(a_H)**2 + np.abs(a_V)**2
    
    # Prevent divide by zero in normalization
    safe_S0 = np.copy(S0)
    safe_S0[safe_S0 == 0] = 1e-9
    
    S1 = (np.abs(a_H)**2 - np.abs(a_V)**2) / safe_S0
    S2 = (2 * np.real(a_H * np.conj(a_V))) / safe_S0
    S3 = (2 * np.imag(a_H * np.conj(a_V))) / safe_S0
    
    # --- SCORING: THE GOLDILOCKS CHECK ---
    max_intensity = np.max(S0)
    
    # Condition 1: CHIP MELTED (Too much Power)
    if max_intensity > 300: 
        title_text.set_text("STATUS: CHIP MELTED (THERMAL OVERLOAD)")
        score_text.set_text("RESONANCE: 0% (FAIL)")
        score_text.set_color('#ff0000') # Red
        # Clear lines to show failure
        line_output.set_data([], [])
        line_output.set_3d_properties([])
        line_sync.set_data([], [])
        fig.canvas.draw_idle()
        return

    # Condition 2: SIGNAL LOST (Too weak or mismatched)
    if max_intensity < 1.0:
        title_text.set_text("STATUS: SIGNAL LOST (TOO WEAK)")
        score_text.set_text("RESONANCE: 0%")
        score_text.set_color('#555555') # Grey
        fig.canvas.draw_idle()
        return

    # Condition 3: CALCULATE RESONANCE
    # We want high correlation AND good dynamic range (not flat saturation)
    raw_correlation = np.abs(np.corrcoef(x_in, S0)[0, 1])
    
    # Hard Mode Penalty: If variance is low (flat line), score drops
    dynamic_range = np.std(S0)
    if dynamic_range < 5:
        penalty_factor = 0.2 # Boring flat line
    else:
        penalty_factor = 1.0
        
    final_score = raw_correlation * 100 * penalty_factor
    
    # Update Texts based on Score
    score_text.set_text(f"RESONANCE: {final_score:.1f}%")
    
    if final_score > 90:
        title_text.set_text("STATUS: SINGULARITY OPEN (STABLE)")
        score_text.set_color('#00ff00') # Green
        line_output.set_color('#00ffcc') # Cyan
    elif final_score > 60:
        title_text.set_text("STATUS: LOCKING IN...")
        score_text.set_color('#ffff00') # Yellow
        line_output.set_color('#ffaa00') # Orange
    else:
        title_text.set_text("STATUS: UNSTABLE / NOISY")
        score_text.set_color('#ff0055') # Pink/Red
        line_output.set_color('#ff0055')

    # Update Plots
    line_output.set_data(S1, S2)
    line_output.set_3d_properties(S3)
    
    line_sync.set_data(x_in, S0)
    # Auto-scale Sync plot y-axis to prevent it from jumping too much
    ax_sync.set_ylim(0, max(50, max_intensity * 1.1))
    ax_sync.set_xlim(np.min(x_in), np.max(x_in))

    fig.canvas.draw_idle()

# Connect sliders to update function
s_pump.on_changed(update)
s_gamma.on_changed(update)
s_chi.on_changed(update)

# Initial Run
update(0)

# Show the Interface
plt.show()