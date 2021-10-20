import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def circuit2unitary(qc: QuantumCircuit) -> list:
    for gate in qc.data:
        if hasattr(gate[0], 'to_matrix'):
            print(gate[0].to_matrix())