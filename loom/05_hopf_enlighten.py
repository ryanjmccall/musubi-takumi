import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.integrate import odeint

"""
UI DASHBOARD CONFIGURATIONS:
-------------------------------------------------------
1. LOVE (Resonance)
   - Set Coupling  ~ 15.0
   - Set Friction  ~ 0.2
   * Result: The 3D path spirals towards the equator (sharing energy), 
     but slowly collapses due to friction.

2. JOY (Coherence) 
   - Set Coupling  = 0.0
   - Set Friction  < 0.05
   * Result: The 3D path stays locked near a Pole (Self-contained). 
     It is stable but isolated.

3. ENLIGHTENMENT (Superconductivity)
   - Set Coupling  > 15.0
   - Set Friction  = 0.0
   * Result: The 3D path forms a perfect, closed loop (a knot) on the sphere.
     The system cycles through states of self and other infinitely with zero loss.
"""

# --- 1. System Definition ---
def system(state, t, m1, m2, k1, k2, c1, c2, k_couple):
    x1, v1, x2, v2 = state
    dx1dt = v1
    dv1dt = (-k1*x1 - c1*v1 + k_couple*(x2 - x1)) / m1
    dx2dt = v2
    dv2dt = (-k2*x2 - c2*v2 - k_couple*(x2 - x1)) / m2
    return [dx1dt, dv1dt, dx2dt, dv2dt]

# --- 2. The Hopf Map Function ---
def hopf_map(x1, v1, x2, v2):
    # Convert state to complex numbers
    z1 = x1 + 1j * v1
    z2 = x2 + 1j * v2
    
    # Normalize to project onto S3 (The Total Energy Shell)
    norm = np.sqrt(np.abs(z1)**2 + np.abs(z2)**2) + 1e-9
    z1 /= norm
    z2 /= norm
    
    # Hopf Map formula: S3 -> S2 (Projecting 4D complex state to 3D real coordinates)
    # This shows the RELATIONSHIP between the two systems.
    X = 2 * (z1 * z2.conjugate()).real
    Y = 2 * (z1 * z2.conjugate()).imag
    Z = np.abs(z1)**2 - np.abs(z2)**2
    return X, Y, Z

# --- 3. Simulation Parameters ---
t = np.linspace(0, 40, 2000)
state0 = [2.0, 0.0, -1.0, 0.0] 
m1_init, m2_init = 1.0, 1.0
k1_init, k2_init = 10.0, 12.0
c_init = 0.2
k_couple_init = 0.0

# --- 4. Visualization Setup ---
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Subplot 1: Phase Space (2D)
ax_phase = fig.add_subplot(1, 2, 1)
ax_phase.set_title("Individual Self (Phase Space)")
ax_phase.set_xlabel("Position")
ax_phase.set_ylabel("Velocity")
ax_phase.set_xlim(-3, 3)
ax_phase.set_ylim(-6, 6)
ax_phase.grid(True)
line1_phase, = ax_phase.plot([], [], 'b-', alpha=0.5, label='Self')
line2_phase, = ax_phase.plot([], [], 'r-', alpha=0.5, label='Other')
point1, = ax_phase.plot([], [], 'bo')
point2, = ax_phase.plot([], [], 'ro')
ax_phase.legend()

# Subplot 2: The Hopf Fibration (3D)
ax_hopf = fig.add_subplot(1, 2, 2, projection='3d')
ax_hopf.set_title("Global Topology (The Hopf Map)")
ax_hopf.set_xlim(-1, 1)
ax_hopf.set_ylim(-1, 1)
ax_hopf.set_zlim(-1, 1)

# Draw the wireframe sphere (The "World" of interaction)
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sphere = np.cos(u)*np.sin(v)
y_sphere = np.sin(u)*np.sin(v)
z_sphere = np.cos(v)
ax_hopf.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.1)

# The trajectory on the sphere
line_hopf, = ax_hopf.plot([], [], [], 'g-', linewidth=2, label='State Topology')
point_hopf, = ax_hopf.plot([], [], [], 'go')

# --- 5. Controls ---
ax_couple = plt.axes([0.2, 0.1, 0.6, 0.03])
ax_damping = plt.axes([0.2, 0.05, 0.6, 0.03])
s_couple = Slider(ax_couple, 'Coupling (Love)', 0.0, 25.0, valinit=k_couple_init)
s_damping = Slider(ax_damping, 'Friction (Resistance)', 0.0, 1.0, valinit=c_init)

# --- 6. Logic ---
solution = odeint(system, state0, t, args=(m1_init, m2_init, k1_init, k2_init, c_init, c_init, k_couple_init))

def update(val):
    global solution
    solution = odeint(system, state0, t, args=(m1_init, m2_init, k1_init, k2_init, s_damping.val, s_damping.val, s_couple.val))

s_couple.on_changed(update)
s_damping.on_changed(update)

def animate(i):
    # 2D Data
    x1, v1 = solution[:i+1, 0], solution[:i+1, 1]
    x2, v2 = solution[:i+1, 2], solution[:i+1, 3]
    
    # 3D Hopf Data
    X, Y, Z = hopf_map(x1, v1, x2, v2)
    
    # Update 2D
    line1_phase.set_data(x1, v1)
    line2_phase.set_data(x2, v2)
    point1.set_data([solution[i, 0]], [solution[i, 1]])
    point2.set_data([solution[i, 2]], [solution[i, 3]])
    
    # Update 3D - Keep only recent history to avoid clutter, or full history for shape
    # Showing full history to see the "Knot" form
    line_hopf.set_data(X, Y)
    line_hopf.set_3d_properties(Z)
    point_hopf.set_data([X[-1]], [Y[-1]])
    point_hopf.set_3d_properties([Z[-1]])
    
    return line1_phase, line2_phase, line_hopf, point1, point2, point_hopf

ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=False)
plt.show()