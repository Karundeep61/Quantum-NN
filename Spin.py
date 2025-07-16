from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt

# Create a quantum circuit and apply a Hadamard gate to create superposition
qc = QuantumCircuit(1)
qc.h(0)

# Simulate the statevector to visualize the superposition on the Bloch sphere
state = Statevector.from_instruction(qc)
plot_bloch_multivector(state)
plt.title("Bloch Sphere: Superposition |+⟩")
plt.show()

# Add a measurement
qc.measure_all()

# Use AerSimulator (modern replacement for qasm_simulator)
backend = AerSimulator()
job = backend.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()

# Plot measurement histogram
plot_histogram(counts)
plt.title("Measurement Outcomes")
plt.show()
