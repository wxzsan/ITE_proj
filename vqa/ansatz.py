import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.opflow import Z, X, Y, I, StateFn, CircuitStateFn, SummedOp, ListOp
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian

#Circuit imports
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
def generate_ansatz(k, dim, x = 'zero'):
    num_vars = k*dim + 2*(dim - 1)*k + k*dim*3
    # print(num_vars)
    # dim 是qubit的个数的全局变量
    qr = QuantumRegister(dim, name="q")
    qc = QuantumCircuit(qr)
    params = ParameterVector('theta', length=num_vars)
    it = iter(params)
    if x != 'zero':
        qc.initialize(x, qr)
    for j in range(k):
        for i in range(dim):
            qc.ry(next(it), qr[i])
        qc.barrier()
        for i in range(dim - 1):
            qc.cx(qr[i], qr[i + 1])
        # two qubit
        qc.barrier()
        for i in range(dim-1):
            qc.ryy(next(it), qr[i], qr[i + 1])
            qc.rxx(next(it), qr[i], qr[i + 1])
        qc.barrier()
        for i in range(dim):
            qc.rz(next(it), qr[i])
            qc.ry(next(it), qr[i])
            qc.rz(next(it), qr[i])
        qc.barrier()
    # qc.unitary(eigVec, range(dim))
    # qc.measure(qr, cr)
    # qc.draw()
    return (qc,params)