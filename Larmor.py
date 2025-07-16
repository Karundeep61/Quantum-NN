import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# --- Parameters ---
omega = 2 * np.pi
steps = 100
t_total = 4
dt = t_total / steps

# --- Initial State: |+⟩ ---
qc = QuantumCircuit(1)
qc.h(0)
initial_state = Statevector.from_instruction(qc)

# --- Helper: Get Bloch coords ---
def bloch_coords(state):
    a, b = state.data
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a)**2 - np.abs(b)**2
    return [x, y, z]

# --- Create trajectory ---
x_vals, y_vals, z_vals = [], [], []
for frame in range(steps):
    angle = omega * dt * frame
    step_circ = QuantumCircuit(1)
    step_circ.rz(angle, 0)
    evolved = initial_state.evolve(step_circ)
    x, y, z = bloch_coords(evolved)
    x_vals.append(x)
    y_vals.append(y)
    z_vals.append(z)

# --- Plot using Plotly ---
fig = go.Figure()

# Bloch Sphere Grid
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
sphere_x = np.outer(np.cos(u), np.sin(v))
sphere_y = np.outer(np.sin(u), np.sin(v))
sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
fig.add_surface(x=sphere_x, y=sphere_y, z=sphere_z, opacity=0.2, colorscale='Greys')

# Quantum State Vector Animation
for i in range(steps):
    fig.add_trace(go.Scatter3d(
        x=[0, x_vals[i]],
        y=[0, y_vals[i]],
        z=[0, z_vals[i]],
        mode='lines+markers',
        line=dict(color='red', width=6),
        marker=dict(size=4),
        name=f"t={round(i*dt,2)}s",
        visible=(i==0)
    ))

# Slider for animation
steps_slider = []
for i in range(steps):
    step = dict(method="update",
                args=[{"visible": [False]*steps}],
                label=f"{i}")
    step["args"][0]["visible"][i] = True
    steps_slider.append(step)

sliders = [dict(active=0, pad={"t": 50}, steps=steps_slider)]
fig.update_layout(title="Larmor Precession: Bloch Sphere",
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                  sliders=sliders)

fig.show()
