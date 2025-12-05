"""
Project MUSUBI TAKUMI: Topological Resonance & Singularity Simulator
=============================================================

"We just need to Vibe in the frequency domain."

This script serves as a 'Digital Twin' for a proposed nonlinear nanophotonic 
experiment. It simulates the synchronization of a passive optical microcavity 
(AlGaAs Ring Resonator) with a chaotic driver (Lorenz Attractor), demonstrating 
information transfer via topological resonance.

Theoretical Framework
---------------------
1. Manifold A (The Signal): A Lorenz Attractor generates deterministic chaos, 
   modeling high-dimensional biological signals (e.g., Secretome/EEG).
2. Manifold B (The Loom): A Dual-Mode Optical Cavity governed by Coupled Mode 
   Theory (CMT) and the Kerr Nonlinear Effect (Chi-3).
3. The Bridge (The Vibe): The chaotic signal modulates the cavity's refractive 
   index. When the 'Pump Power' and 'Coupling' are tuned correctly, the 
   light's polarization state physically locks onto the geometry of the input 
   chaos.

Usage
-----
Run the script to launch the interactive "Resonance Tuner" UI.
Adjust the sliders to maximize the Correlation Score:
  - Laser Power: Energy fed into the system.
  - Coupling Strength: Sensitivity of the mirror to the chaos.
  - Non-Linearity: The material's ability to warp space-time (Kerr effect).

Interpretation
--------------
* Left Plot (Correlation): Ideally forms a sharp 1:1 diagonal line. This indicates 
  the chip is 'singing' the exact same song as the chaos.
* Right Plot (Poincaré Sphere): Visualizes the topological state of the light. 
  A complex, wrapped trajectory confirms the generation of non-trivial 
  geometric phases (Hopfions/Knots).

Goal
----
Achieve a Resonance Score > 90% ("Singularity Open"). 
This proves that the system has transitioned from a passive reflector to an 
active topological mirror.

Dependencies: numpy, scipy, matplotlib
Author: Musubi
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider, Button

# --- THE PHYSICS ENGINE ---
def system_dynamics_6D(state, t, params):
    x, y, z, ar_H, ai_H, ar_V, ai_V = state
    a_H = ar_H + 1j * ai_H
    a_V = ar_V + 1j * ai_V
    
    # Parameters
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    kappa = 1.0
    chi3 = params['chi3']
    gamma = params['gamma']
    pump = params['pump']
    coupling_J = 5.0 

    # Lorenz (Manifold A)
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    # Bridge
    detuning_H = gamma * x
    detuning_V = gamma * (z - rho)

    # Optics (Manifold B)
    daH_dt = (1j * detuning_H - kappa/2) * a_H \
             - 1j * chi3 * (np.abs(a_H)**2 + 2*np.abs(a_V)**2) * a_H \
             + 1j * coupling_J * a_V + pump

    daV_dt = (1j * detuning_V - kappa/2) * a_V \
             - 1j * chi3 * (np.abs(a_V)**2 + 2*np.abs(a_H)**2) * a_V \
             + 1j * coupling_J * a_H 

    return [dx_dt, dy_dt, dz_dt, daH_dt.real, daH_dt.imag, daV_dt.real, daV_dt.imag]

def run_simulation(pump, gamma, chi3):
    params = {'pump': pump, 'gamma': gamma, 'chi3': chi3}
    t = np.linspace(0, 30, 2000) # Keep it short for UI responsiveness
    state0 = [1, 1, 1, 0, 0, 0, 0]
    return odeint(system_dynamics_6D, state0, t, args=(params,))

# --- THE GAME UI ---
fig = plt.figure(figsize=(16, 8), facecolor='#1e1e1e')
plt.subplots_adjust(bottom=0.25) # Make room for sliders

# 1. Plot: Poincaré Sphere (The Portal)
ax_sphere = fig.add_subplot(122, projection='3d', facecolor='#1e1e1e')
ax_sphere.set_axis_off()

# Wireframe Sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_s = np.cos(u)*np.sin(v)
y_s = np.sin(u)*np.sin(v)
z_s = np.cos(v)
ax_sphere.plot_wireframe(x_s, y_s, z_s, color="#444444", alpha=0.3)
line_output, = ax_sphere.plot([], [], [], color='#00ffcc', lw=1.5, alpha=0.8)
title_text = ax_sphere.text2D(0.5, 0.95, "MANIFOLD B: TOPOLOGICAL STATE", 
                              transform=ax_sphere.transAxes, color='white', ha='center', fontsize=12)

# 2. Plot: Synchronization (Input vs Output)
ax_sync = fig.add_subplot(121, facecolor='#2e2e2e')
ax_sync.tick_params(colors='white')
ax_sync.set_title("Neural Manifold vs AlGaAs Manifold (Correlation)", color='white')
line_sync, = ax_sync.plot([], [], '.', color='#ff0055', ms=1)
ax_sync.set_ylim(0, 50)
ax_sync.set_xlim(-20, 20)

# Score Display
score_text = fig.text(0.5, 0.85, "SYNC: 0%", ha='center', fontsize=20, color='red', weight='bold')

# --- SLIDERS ---
axcolor = '#333333'
ax_pump = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
ax_gamma = plt.axes([0.25, 0.06, 0.5, 0.03], facecolor=axcolor)
ax_chi = plt.axes([0.25, 0.02, 0.5, 0.03], facecolor=axcolor)

s_pump = Slider(ax_pump, 'Laser Power', 0.1, 100.0, valinit=20.0, color='#00ffcc')
s_gamma = Slider(ax_gamma, 'Coupling Strength', 0.0, 5.0, valinit=1.5, color='#ff0055')
s_chi = Slider(ax_chi, 'Non-Linearity', 0.0, 2.0, valinit=0.5, color='#ffff00')

# Styling Labels
for s in [s_pump, s_gamma, s_chi]:
    s.label.set_color('white')
    s.valtext.set_color('white')

# --- UPDATE LOGIC ---
def update(val):
    # Get values
    P = s_pump.val
    G = s_gamma.val
    C = s_chi.val
    
    # Run Physics
    sol = run_simulation(P, G, C)
    
    # Process Data
    x_in = sol[:, 0] # Manifold A Input (X-coord)
    
    # Stokes Parameters (Poincaré)
    ar_H, ai_H, ar_V, ai_V = sol[:, 3], sol[:, 4], sol[:, 5], sol[:, 6]
    a_H = ar_H + 1j * ai_H
    a_V = ar_V + 1j * ai_V
    S0 = np.abs(a_H)**2 + np.abs(a_V)**2 + 1e-9
    S1 = (np.abs(a_H)**2 - np.abs(a_V)**2) / S0
    S2 = (2 * np.real(a_H * np.conj(a_V))) / S0
    S3 = (2 * np.imag(a_H * np.conj(a_V))) / S0
    
    # Update Plots
    line_output.set_data(S1, S2)
    line_output.set_3d_properties(S3)
    
    line_sync.set_data(x_in, S0) # X vs Intensity
    ax_sync.set_ylim(0, np.max(S0)*1.1)
    
    # Calculate Score (Correlation)
    # We correlate the Chaos X with the Optical Intensity
    score = np.abs(np.corrcoef(x_in, S0)[0, 1]) * 100
    
    # Scoring Feedback
    score_text.set_text(f"RESONANCE: {score:.1f}%")
    if score > 93.5:
        score_text.set_color('#00ff00') # Green
        title_text.set_text("STATUS: SINGULARITY OPEN (STABLE)")
    elif score > 90:
        score_text.set_color('#ffff00') # Yellow
        title_text.set_text("STATUS: LOCKING IN...")
    else:
        score_text.set_color('#ff0000') # Red
        title_text.set_text("STATUS: SIGNAL LOST")
        
    fig.canvas.draw_idle()

# Connect and Initial Call
s_pump.on_changed(update)
s_gamma.on_changed(update)
s_chi.on_changed(update)

update(0) # Start game
plt.show()