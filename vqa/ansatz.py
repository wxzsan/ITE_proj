import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.opflow import Z, X, Y, I, CX, StateFn, CircuitStateFn, SummedOp, ListOp
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import csgraph

#Circuit imports
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression


class QuantumCircuitUnitary:
    def __init__(self, qcnt):
        self.qcnt = qcnt
        self.gate_list = []
    
    def rx(self, i):
        #之后可以换成to_spmatrix
        self.gate_list.append([((I^i)^X^(I^(self.qcnt-i-1))).to_matrix(), True])
    def ry(self, i):
        self.gate_list.append([((I^i)^Y^(I^(self.qcnt-i-1))).to_matrix(), True])
    def rz(self, i):
        self.gate_list.append([((I^i)^Z^(I^(self.qcnt-i-1))).to_matrix(), True])
    def cx(self, i):
        self.gate_list.append([((I^i)^CX^(I^(self.qcnt-i-2))).to_matrix(), False])
    def rxx(self, i, j):
        if i == j:
            print("apply rxx to one qubit")
        elif i < j:
            self.gate_list.append([((I^i)^X^(I^(j-i-1))^X^(I^(self.qcnt-j-1))).to_matrix(), True])
        else:
            self.gate_list.append([((I^j)^X^(I^(i-j-1))^X^(I^(self.qcnt-i-1))).to_matrix(), True])
    def ryy(self, i, j):
        if i == j:
            print("apply ryy to one qubit")
        elif i < j:
            self.gate_list.append([((I^i)^Y^(I^(j-i-1))^Y^(I^(self.qcnt-j-1))).to_matrix(), True])
        else:
            self.gate_list.append([((I^j)^Y^(I^(i-j-1))^Y^(I^(self.qcnt-i-1))).to_matrix(), True])
    def rzz(self, i, j):
        if i == j:
            print("apply rzz to one qubit")
        elif i < j:
            self.gate_list.append([((I^i)^Z^(I^(j-i-1))^Z^(I^(self.qcnt-j-1))).to_matrix(), True])
        else:
            self.gate_list.append([((I^j)^Z^(I^(i-j-1))^Z^(I^(self.qcnt-i-1))).to_matrix(), True])
class QuantumCircuitUnitarySparse:
    def __init__(self, qcnt):
        self.qcnt = qcnt
        self.gate_list = []
    
    def rx(self, i):
        #之后可以换成to_spmatrix
        self.gate_list.append([((I^i)^X^(I^(self.qcnt-i-1))).to_spmatrix(), True])
    def ry(self, i):
        self.gate_list.append([((I^i)^Y^(I^(self.qcnt-i-1))).to_spmatrix(), True])
    def rz(self, i):
        self.gate_list.append([((I^i)^Z^(I^(self.qcnt-i-1))).to_spmatrix(), True])
    def cx(self, i):
        self.gate_list.append([((I^i)^CX^(I^(self.qcnt-i-2))).to_spmatrix(), False])
    def rxx(self, i, j):
        if i == j:
            print("apply rxx to one qubit")
        elif i < j:
            self.gate_list.append([((I^i)^X^(I^(j-i-1))^X^(I^(self.qcnt-j-1))).to_spmatrix(), True])
        else:
            self.gate_list.append([((I^j)^X^(I^(i-j-1))^X^(I^(self.qcnt-i-1))).to_spmatrix(), True])
    def ryy(self, i, j):
        if i == j:
            print("apply ryy to one qubit")
        elif i < j:
            self.gate_list.append([((I^i)^Y^(I^(j-i-1))^Y^(I^(self.qcnt-j-1))).to_spmatrix(), True])
        else:
            self.gate_list.append([((I^j)^Y^(I^(i-j-1))^Y^(I^(self.qcnt-i-1))).to_spmatrix(), True])
    def rzz(self, i, j):
        if i == j:
            print("apply rzz to one qubit")
        elif i < j:
            self.gate_list.append([((I^i)^Z^(I^(j-i-1))^Z^(I^(self.qcnt-j-1))).to_spmatrix(), True])
        else:
            self.gate_list.append([((I^j)^Z^(I^(i-j-1))^Z^(I^(self.qcnt-i-1))).to_spmatrix(), True])

def generate_ansatz_qiskit(k, dim, x = 'zero'):
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
        #qc.barrier()
        for i in range(dim - 1):
            qc.cx(qr[i], qr[i + 1])
        # two qubit
        #qc.barrier()
        for i in range(dim-1):
            qc.ryy(next(it), qr[i], qr[i + 1])
            qc.rxx(next(it), qr[i], qr[i + 1])
        #qc.barrier()
        for i in range(dim):
            qc.rz(next(it), qr[i])
            qc.ry(next(it), qr[i])
            qc.rz(next(it), qr[i])
        #qc.barrier()
    # qc.unitary(eigVec, range(dim))
    # qc.measure(qr, cr)
    # qc.draw()
    return (qc,params)

def generate_ansatz(k, dim):
    qc = QuantumCircuitUnitary(dim)
    for j in range(k):
        for i in range(dim):
            qc.ry(i)
        for i in range(dim - 1):
            qc.cx(i)
        # two qubit
        for i in range(dim-1):
            qc.ryy(i, i+1)
            qc.rxx(i, i+1)
        for i in range(dim):
            qc.rz(i)
            qc.ry(i)
            qc.rz(i)
    return qc

def generate_ansatz_sparse(k, dim):
    qc = QuantumCircuitUnitarySparse(dim)
    for j in range(k):
        for i in range(dim):
            qc.ry(i)
        for i in range(dim - 1):
            qc.cx(i)
        # two qubit
        for i in range(dim-1):
            qc.ryy(i, i+1)
            qc.rxx(i, i+1)
        for i in range(dim):
            qc.rz(i)
            qc.ry(i)
            qc.rz(i)
    return qc