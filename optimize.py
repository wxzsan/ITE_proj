from vqa.ansatz import *
from vqa.hamiltonian import generate_hamiltonian
import numpy as np
from vqa.gradient import *
from vqa.transform import *
from scipy.optimize import *

dim = 3
num_layer = 2

qc_qiskit, params = generate_ansatz_qiskit(num_layer,dim)
num_vars = len(params)
qc = generate_ansatz(num_layer, dim)
qc_sparse = generate_ansatz_sparse(num_layer, dim)

hamiltonian_matrix_sparse = generate_hamiltonian(dim).eval().to_spmatrix()
hamiltonian_matrix = generate_hamiltonian(dim).eval().to_matrix()


def objective_function(x):
    #values_dict = dict(zip(params, x))
    values_dict = x
    statevector = forward(qc, values_dict)
    return np.real(statevector@hamiltonian_matrix@statevector.conj())

# sparse version can achieve speed-up when dimension of the system is large 
def objective_function_sparse(x):
    #values_dict = dict(zip(params, x))
    values_dict = x
    statevector = forward_sparse(qc_sparse, values_dict)
    return np.real(statevector@hamiltonian_matrix_sparse@statevector.conj())

def gradient_function(x):
    values_dict = x
    eps = 1e-6
    current_value = objective_function(x)
    gradient = np.zeros(num_vars)
    for i in range(num_vars):
        x[i] += eps
        gradient[i] = np.real((objective_function(x)-current_value))/eps
        x[i] -= eps
    return gradient

def natural_gradient_function(x):
    eta = 0.2
    gradient = np.linalg.inv(get_qfi(qc, x) + eta*np.identity(num_vars))@gradient_function(x)
    return np.real(gradient)


def gradient_function_sparse(x):
    #values_dict = dict(zip(params, x))
    values_dict = x
    eps = 1e-6
    eta = 1e-1
    current_value = objective_function_sparse(x)
    gradient = np.zeros(num_vars)
    for i in range(num_vars):
        x[i] += eps
        gradient[i] = np.real((objective_function_sparse(x)-current_value))/eps
        x[i] -= eps
    gradient = np.linalg.inv(get_qfi_sparse(qc_sparse, values_dict) + eta*np.identity(num_vars))@gradient
    gradient = np.real(gradient)
    return gradient

def gradient_descent(x0, f, f_directtion, f_gradient, adaptative = False):
    x_i = x0
    all_x_i = list()
    all_f_i = list()
    for i in range(100):
        all_x_i.append(x_i)
        all_f_i.append(f(x_i))
        dx_i = f_directtion(x_i)
        gradient_i = f_gradient(x_i)
        if(np.linalg.norm(gradient_i) < 1e-4):
            print("optimization converges")
            break
        print(all_f_i[-1])
        if adaptative:
            # Compute a step size using a line_search to satisfy the Wolf
            # conditions
            step = line_search(f, f_gradient,
                                x_i, -dx_i, gradient_i,
                                c2=.05)
            step = step[0]
            if step is None:
                step = 0
        else:
            step = 1
        x_i += -step*dx_i
        if np.abs(all_f_i[-1]) < 1e-16:
            break
    return all_x_i, all_f_i
def call_back(x):
    print("current value: ", objective_function(x))

x = np.random.rand(num_vars)

res = gradient_descent(x, objective_function, natural_gradient_function, gradient_function, adaptative = True)
print(len(res[0]))