import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import warnings

# Ignore deprecation warnings temporarily
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch Tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# QNN Circuit
num_qubits = 2
feature_map = ZZFeatureMap(feature_dimension=num_qubits)
ansatz = RealAmplitudes(num_qubits, reps=1)

qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

# Estimator-based QNN (V1, soon to be deprecated)
qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

# PyTorch Connector
model = TorchConnector(qnn)

# PyTorch model
class QuantumClassifier(nn.Module):
    def __init__(self, quantum_model):
        super().__init__()
        self.qnn = quantum_model
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        x = self.qnn(x)
        x = self.linear(x)
        return torch.sigmoid(x)

qc_model = QuantumClassifier(model)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(qc_model.parameters(), lr=0.1)

# Train
for epoch in range(30):
    optimizer.zero_grad()
    output = qc_model(X_train)
    loss = criterion(output.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    y_pred = qc_model(X_test).squeeze().round()
    acc = (y_pred == y_test).float().mean()
    print(f"\nTest Accuracy: {acc.item() * 100:.2f}%")

# Plot
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title("Quantum Classifier Prediction (EstimatorQNN)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
