from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city, plot_histogram, circuit_drawer
import matplotlib.pyplot as plt

# Step 1: Create Bell State Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Step 2: Get final statevector
state = Statevector.from_instruction(qc)
print("Bell Statevector:")
print(state)

# Step 3: Plot the circuit
fig = qc.draw('mpl')
fig.savefig("bell_circuit.png")
plt.show()

# Step 4: Plot probability histogram (derived manually from statevector)
probs = state.probabilities_dict()
print("\nProbabilities:")
print(probs)

plot_histogram(probs)
plt.title("Bell State |00> + |11> Measurement Probabilities")
plt.show()
