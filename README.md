# Quantum-NN
🧠 Simulations in quantum computing using Qiskit: Bloch sphere visualizations, Bell state entanglement, and a hybrid Quantum Neural Network (QNN) — all running locally with classical backends. No real quantum hardware required.

# 🧠 Quantum Computing with Qiskit: Simulations, Entanglement & QNNs

This repository contains hands-on simulations and quantum machine learning experiments built using **IBM's Qiskit** framework. The goal is to explore fundamental quantum principles such as **superposition**, **entanglement**, **Larmor precession**, and **quantum neural networks**, with full local simulation support.

> 🧪 All simulations run locally using classical simulators — no quantum hardware required.

---

## 📌 Project Highlights

### 🌀 1. Bloch Sphere & Superposition
- Simulates the **state of a single qubit** using the Bloch sphere representation.
- Demonstrates the effect of basic quantum gates like `H`, `X`, `Y`, and `Z`.
- Visualizes how the qubit behaves under different unitary operations.

> Example: `bloch_visualization.py`

---

### 🧲 2. Larmor Precession (Spin in Magnetic Field)
- Simulates the **Larmor precession** of a spin-½ particle (e.g., electron) in a static magnetic field using a time-evolution operator.
- Based on the Hamiltonian:  
  \[
  H = -\frac{\omega}{2} \sigma_z \quad \Rightarrow \quad U(t) = e^{i\omega t \sigma_z / 2}
  \]
- Shows the continuous precession of the qubit's state on the Bloch sphere over time.

> Example: `larmor_precession.py`

---

### 🔗 3. Quantum Entanglement (Bell State)
- Constructs a **Bell state** using two qubits.
- Shows how `H` and `CNOT` gates generate an entangled quantum state.
- Uses the `Statevector` and `DensityMatrix` tools from Qiskit to analyze the reduced states of individual qubits.
- Attempts to plot the Bloch vector of entangled qubits (results in mixed states with no well-defined Bloch vector).

> Example: `entanglement.py`

---

### 🧬 4. Quantum Neural Network (QNN)
- Implements a **hybrid quantum-classical neural network** using Qiskit's `EstimatorQNN`.
- Simulates binary classification using fake data.
- Uses a `ZZFeatureMap` to encode classical input data into entangled quantum states.
- Integrates with **PyTorch** to train the model over 30 epochs.

> Example: `qnn.py`

---

## ⚙️ Setup Instructions

### 🔧 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
