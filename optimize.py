from vqa.ansatz import generate_ansatz
from vqa.hamiltonian import generate_hamiltonian
from qiskit.opflow import I, X, Y, Z, OperatorStateFn, StateFn, CircuitStateFn
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian
import numpy as np
import matplotlib.pyplot as plt
from vqa.gradient import *
import time
import qiskit.algorithms.optimizers
from scipy.optimize import minimize
import time

dim = 10
num_layer = 1

qc, params = generate_ansatz(num_layer,dim)
num_vars = len(params)

hamiltonian_matrix = generate_hamiltonian(dim).eval().to_matrix()

def objective_function(x):
    values_dict = dict(zip(params, x))
    statevector = forward(qc, values_dict)
    return statevector@hamiltonian_matrix@statevector.conj()

def gradient_function(x):
    values_dict = dict(zip(params, x))
    eps = 1e-6
    eta = 1e-1
    current_value = objective_function(x)
    gradient = np.zeros(num_vars)
    for i in range(num_vars):
        x[i] += eps
        gradient[i] = np.real((objective_function(x)-current_value))/eps
        x[i] -= eps
    gradient = np.linalg.inv(get_qfi(qc, values_dict) + eta*np.identity(num_vars))@gradient
    gradient = gradient
    return gradient

def gradient_function_param_shift(x):
    values_dict = dict(zip(params, x))
    eta = 1e-1
    gradient = np.zeros(num_vars)
    for i in range(num_vars):
        tmp = x[i]
        x[i] += np.pi/2
        gradient[i] = objective_function(x)
        x[i] -= np.pi
        gradient[i] = np.real(gradient[i] - objective_function(x))/2
        x[i] = tmp

    # gradient = np.linalg.inv(get_qfi(qc, values_dict) + eta*np.identity(num_vars))@gradient
    gradient = gradient
    return gradient

def call_back(x):
    print("current value: ", objective_function(x))

x = np.random.rand(num_vars)
res = minimize(fun = objective_function, x0 = x, method = 'CG', jac = gradient_function, callback = call_back)