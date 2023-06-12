import numpy as np
from qiskit import QuantumCircuit, assemble
from qiskit.providers.aer import AerSimulator
from collections.abc import Iterable
from qiskit.quantum_info import partial_trace
from qiskit.quantum_info.operators import Operator
from Utils import Adjoint as Ad
from Entanglement.WernerUtils import witness_Qk

class MyQuantumCircuit():
    
    def __init__(self, qc:QuantumCircuit, shots = 1024) -> None:
        self.__measured = False
        self.shots = shots
        self.backend = AerSimulator()
        self.set_circuit(qc)
        

    def set_circuit(self, qc: QuantumCircuit):
        self.circuit = qc

    def run(self):
        if self.__measured == False:
            self.circuit.measure_all()
            self.__measured = True
        job = self.backend.run(self.circuit, shots = self.shots)
        counts = job.result().get_counts()
        self.counts = counts
        return counts
    
    def measure(self, q_lst, c_lst):
        self.circuit.measure(q_lst, c_lst)
        # def prepare(self):
        
            # self.circuit = transpile(self.circuit, self.backend)

    def eval_expect(self, i):
        if isinstance(i, int):
            lst = [(-1)**int(key[i]) * self.counts[key] for key in self.counts.keys() ]
        elif isinstance(i, Iterable):
            lst = [(-1)**sum([int(key[j]) for j in i ]) * self.counts[key] for key in self.counts.keys()]
        return sum(lst)/self.shots


class ParaQuCircuit(MyQuantumCircuit):
    def __init__(self, shots=1024) -> None:
        super().__init__(shots)
    
    def set_circuit(self, qc: QuantumCircuit):
        super().set_circuit(qc)
        super().prepare()
        self.parameters = self.circuit.parameters
        self.num_para = self.circuit.num_parameters
    
    def bind_paras(self, parameter):
        value_dict = dict(zip(self.parameters, parameter))
        qc = MyQuantumCircuit(self.circuit.bind_parameters(value_dict), self.shots)
        qc.run()
        return qc
    
    def rand_paras(self):
        return np.random.random(self.num_para) *2*np.pi


class QNN(MyQuantumCircuit):
    def __init__(self, qc: QuantumCircuit, shots=1024) -> None:
        super().__init__(qc, shots)
        # self.circuit.measure_all()
        self.parameters = self.circuit.parameters
        self.num_para = self.circuit.num_parameters

    def encode(self, strategy, data_lst):
        batch = list()
        labels = list()
        if strategy == 'initialize':
            for data in data_lst:
                encoding = QuantumCircuit(self.circuit.num_qubits, self.circuit.num_clbits)
                encoding.initialize(data[0])
                batch.append( encoding.compose(self.circuit))
                labels.append(data[1])
        self.batch = batch
        self.labels = labels

    def eval_expect(self, counts_lst):
            lst = [(-1)**sum([int(j) for j in key ]) * counts_lst[key] for key in counts_lst.keys()]
            return sum(lst)/self.shots
    
    def run_batch(self, parameter):
        self.assigned_para = dict(zip(self.parameters, parameter))
        qobj = assemble([circ.bind_parameters(self.assigned_para) for circ in self.batch])
        counts = self.backend.run(qobj, shots = self.shots).result().get_counts()
    
        self.expects = [self.eval_expect(count) for count in counts]
        return self.expects
    
    def rand_paras(self):
        return np.random.random(self.num_para) *2*np.pi


class QNN_for_Werner():

    def __init__(self, num_part, num_layer, num_ancilla) -> None:
        """Below has another version of __init__"""
        self.num_part = num_part
        self.num_ancilla = num_ancilla
        self.circuit = QuantumCircuit( 2*self.num_part + num_ancilla )
        self.circuit.compose(RealAmplitudes(
            2*self.num_part + num_ancilla,  reps=num_layer #, entanglement= 'circular'
            ), inplace=True)
        
        eigval = np.linalg.eigvals(witness_Qk(num_part))
        self.observable = np.diag(eigval)

    def get_O_tilde(self, para):
        val_dic = dict(zip(self.circuit.parameters, para))
        sign_qc = self.circuit.bind_parameters(val_dic)
        U = np.array(Operator(sign_qc) )
        self.O_tilde = Ad(U, self.observable)
        return self.O_tilde

    # def get_H(self):
    #     H = partial_trace(self.O_tilde, 
    #         range(2*self.num_part, 2*self.num_part + self.num_ancilla))
    #     return np.array(H)

    def rand_para(self):
        return (np.random.random(self.circuit.num_parameters)-0.5)* 2*np.pi
    
    # def get_expects(self, rho, para):
    #     self.get_O_tilde(para)
    #     H= self.get_H()
    #     return trace(rho @ H).real

    # def __init__(self, num_part, num_layer, num_ancilla) -> None:
    #     self.num_part = num_part
    #     self.num_ancilla = num_ancilla
    #     self.circuit = QuantumCircuit( 2*self.num_part + num_ancilla )
    #     self.circuit.compose(RealAmplitudes(
    #         2*self.num_part + num_ancilla,  reps=num_layer
    #         ), inplace=True)

    #     qc = QuantumCircuit( 2*self.num_part)
    #     qc.swap(range(self.num_part), range(self.num_part, 2*self.num_part))
    #     H = np.array(Operator(qc))
    #     eigval = np.linalg.eigvals(H)

    #     if num_ancilla == 0:
    #         self.observable = np.diag(eigval)
    #     elif num_ancilla == 2*num_part:
    #         qc_zi = QuantumCircuit( 2*self.num_part + num_ancilla)
    #         qc_zi.z(range(num_part))
    #         self.observable = np.array(Operator(qc_zi))
    #     else:
    #         N = 2** num_ancilla
    #         self.observable = np.diag(tensor(eigval, np.ones(N) ))




if __name__ == '__main__':
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.circuit import Parameter
    from Utils import rand_state
    theta = Parameter('θ')
    phi = Parameter('Φ')
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cnot(0,1)
    qc.ry(phi, 0)
    qc.measure_all()

    qc = QNN(qc)
    data = [rand_state(2) for _ in range(10)]
    qc.encode('initialize', data)
    print(qc.run_batch(qc.rand_paras()))
