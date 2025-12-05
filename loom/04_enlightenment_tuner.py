import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.integrate import odeint

"""
1. LOVE (Resonance)
   - Set Coupling  ~ 15.0
   - Set Friction  ~ 0.2
   * Result: Systems synchronize frequencies, but struggle against entropy (damping).

2. JOY (Coherence) 
   - Set Coupling  = 0.0
   - Set Friction  < 0.05
   * Result: The individual self stabilizes. 
   Internal turbulence (noise) vanishes; motion becomes laminar.

3. ENLIGHTENMENT (Superconductivity)
   - Set Coupling  > 15.0
   - Set Friction  = 0.0
   * Result: Zero resistance. The distinction between 'self' and 
   'other' collapses into a single, lossless, perpetual wavefunction.
"""

# --- 1. System Definition ---
# This function defines the physics of two coupled oscillators.
def system(state, t, m1, m2, k1, k2, c1, c2, k_couple):
    x1, v1, x2, v2 = state
    
    # Forces on Oscillator 1
    # -k1*x1: Restoring force (internal consistency)
    # -c1*v1: Damping/Friction (loss of joy/energy)
    # +k_couple*(x2 - x1): Coupling force from Oscillator 2 (connection/love)
    dx1dt = v1
    dv1dt = (-k1*x1 - c1*v1 + k_couple*(x2 - x1)) / m1
    
    # Forces on Oscillator 2
    dx2dt = v2
    dv2dt = (-k2*x2 - c2*v2 - k_couple*(x2 - x1)) / m2
    
    return [dx1dt, dv1dt, dx2dt, dv2dt]

# --- 2. Simulation Parameters ---
t = np.linspace(0, 40, 2000) # Time vector
state0 = [2.0, 0.0, -1.0, 0.0] # Initial states (x1, v1, x2, v2)

# Default parameters (Initial state: Uncoupled, with some friction)
m1_init, m2_init = 1.0, 1.0
k1_init, k2_init = 10.0, 12.0 # Slightly different natural frequencies
c_init = 0.2 # Initial damping
k_couple_init = 0.0 # Initially uncoupled

# --- 3. Visualization Setup ---
# --- 3. Visualization Setup ---
fig, ax_phase = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.25)

# Phase Space Plot (v vs x) - Shows the "shape" of the state
ax_phase.set_title("Phase Space Trajectories (The 'Shape' of Being)")
ax_phase.set_xlabel("Position (x)")
ax_phase.set_ylabel("Velocity (v)")
ax_phase.set_xlim(-3, 3)
ax_phase.set_ylim(-6, 6)
ax_phase.grid(True)
line1_phase, = ax_phase.plot([], [], 'b-', label='Oscillator 1 (Self)', alpha=0.6)
line2_phase, = ax_phase.plot([], [], 'r-', label='Oscillator 2 (Other)', alpha=0.6)
point1_phase, = ax_phase.plot([], [], 'bo')
point2_phase, = ax_phase.plot([], [], 'ro')
ax_phase.legend()


# --- 4. Slider Controls ---
ax_couple = plt.axes([0.15, 0.1, 0.7, 0.03])
ax_damping = plt.axes([0.15, 0.05, 0.7, 0.03])

s_couple = Slider(ax_couple, 'Coupling (Love)', 0.0, 20.0, valinit=k_couple_init)
s_damping = Slider(ax_damping, 'Friction (Resistance)', 0.0, 1.0, valinit=c_init)

# --- 5. Animation & Update Logic ---
solution = odeint(system, state0, t, args=(m1_init, m2_init, k1_init, k2_init, c_init, c_init, k_couple_init))

def update(val):
    # Re-run simulation with new slider values
    k_c = s_couple.val
    c = s_damping.val
    global solution
    solution = odeint(system, state0, t, args=(m1_init, m2_init, k1_init, k2_init, c, c, k_c))

s_couple.on_changed(update)
s_damping.on_changed(update)

def animate(i):
    # Update phase space plot
    x1 = solution[:i, 0]
    v1 = solution[:i, 1]
    x2 = solution[:i, 2]
    v2 = solution[:i, 3]
    
    line1_phase.set_data(x1, v1)
    line2_phase.set_data(x2, v2)
    point1_phase.set_data([solution[i, 0]], [solution[i, 1]])
    point2_phase.set_data([solution[i, 2]], [solution[i, 3]])
    
    return line1_phase, line2_phase, point1_phase, point2_phase

ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)

plt.show()