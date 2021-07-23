import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit import Aer



backend = Aer.get_backend('statevector_simulator')

def forward(qc: QuantumCircuit, values_dict: dict) -> np.ndarray:

    qc_assigned = qc.assign_parameters(values_dict)
    job = execute(qc_assigned, backend)
    result = job.result()
    return result.get_statevector(qc_assigned)



def get_qfi(qc, values_dict):
    # first calc par phi/par theta
    eps = 1e-6
    num_params = len(values_dict)
    current_state = forward(qc, values_dict)

    partial_dev = []
    for param in values_dict.keys():
        values_dict[param] += eps
        partial_dev.append((forward(qc, values_dict)-current_state)/eps)
        values_dict[param] -= eps

    qfi = np.zeros((num_params, num_params))

    for i in range(num_params):
        for j in range(num_params):
            qfi[i][j] = 4*np.real(partial_dev[i]@partial_dev[j].conj()-partial_dev[i]@current_state.conj()*current_state@partial_dev[j].conj())
    
    return qfi


