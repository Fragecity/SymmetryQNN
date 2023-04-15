import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler, Gradient, PauliOp
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance

def evaluate_expectation(qcircuit: QuantumCircuit, observable: PauliOp, theta: np.array, sampler) -> float:
    '''
    Calculate the expectation value of a parameterized quantum circuit

    sampler = CircuitSampler(QuantumInstance(AerSimulator())), is a sample simulator
    '''

    measured_qcirc = StateFn(observable, is_measurement=True) @ StateFn(qcircuit)
    pauli_expect = PauliExpectation().convert(measured_qcirc) # with unknown parameters
    value_dict = dict(zip(qcircuit.parameters, theta))
    result = sampler.convert(pauli_expect, params=value_dict).eval()

    return np.real(result)

def grad_i(index:int, theta:np.array, 
        measured_qcirc, qcirc, shifter, sampler) -> float:
    '''
    return the i-th component gradient of <psi| U^+ O U |psi>

    measured_qcirc is a measured circuit. Its an operator
    shifter = Gradient()
    sampler = CircuitSampler(QuantumInstance(AerSimulator())), is a sample simulator
    '''

    grad = shifter.convert(measured_qcirc, params=qcirc.parameters[index])
    value_dict = dict(zip(qcirc.parameters, theta))
    return sampler.convert(grad, value_dict).eval().real