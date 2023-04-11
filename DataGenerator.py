#%% import
import pennylane.numpy as np
import pennylane as qml
import pickle, typing, copy

#%% generate data
def a_data_with_label() -> tuple:
    two_dim_data = (np.random.rand(2) *2 - 1) * np.pi * 0.9
    two_dim_data.requires_grad = False
    label = two_dim_data[0] + two_dim_data[1] > 0 
    return two_dim_data, label

def generate_data(data_num: int) -> list:
    return [a_data_with_label() for _ in range(data_num)]

#%% polynomial encoding

Vector = typing.List[float]
def encoded_data_set(data_num):
    datas = []
    for x, label in generate_data(data_num):
        data = np.append(x,x)
        data = np.append(data, x)
        datas.append((data, label))
    return datas

# %% Matrix operator
def dagger(A) -> np.array:
    A_conj = np.conj(np.array(A))
    return A_conj.T

def Adjoint(U:np.array, O: np.array) -> np.array:
    return U @ O @ dagger(U)

def hs_norm(A):
    """Hilbert Schmidt norm"""
    return np.real(np.trace(dagger(A) @ A))

def generate_freegroup(a_set):
    FreeGroup = list()
    gates_queq = copy.deepcopy(a_set)

    while gates_queq:
        gate = gates_queq.pop()
        if not mat_in_set(gate, FreeGroup):
            FreeGroup.append(gate)
            for s in a_set: gates_queq.append(gate @ s)
        # print(len(FreeGroup))

    return FreeGroup

def mat_in_set(A, Group) -> bool:
     for G in Group:
          if equal(A, G): return True
     return False

def equal(A,B) -> bool:
    B = np.abs(A-B) < 1e-6
    return B.all()
#%% Symmetry
class Symmetry():
    
    def __init__(self) -> None:
        self.group = []
        self.observable = 0

    def _twirling(self, O_tilde):
        SOS_list = [Adjoint(S, O_tilde) for S in self.group ]
        return 1/len(SOS_list) * sum(SOS_list)

    def _get_O_PO(self, U):
        O_tilde = Adjoint(U, self.observable)
        PO = self._twirling(O_tilde)
        return O_tilde, PO
    
    def symmetry_guidance(self, U):
        O_tilde, PO = self._get_O_PO(U)
        return hs_norm(O_tilde - PO) / len(U)

class SwapSymmetry(Symmetry):

    def __init__(self, num_bits) -> None:
        super().__init__()
        self.dev = qml.device('default.qubit', wires=num_bits)
        self.num_bits = num_bits

        @qml.qnode(device=self.dev, interface="torch")
        def swap_circ(i,j):
            qml.SWAP(wires=[i,j])
            return qml.expval(qml.Identity(wires=range(num_bits)))
        
        swap_ele = list(map(qml.matrix(swap_circ), 
                       [0,1,2,3,4,5],
                       [1,2,3,4,5,0]
                       ))
        # qml.matrix(swap_circ)(1,2), qml.matrix(swap_circ)(3,5)
        # self.group = [ qml.matrix(qml.Identity(wires=range(num_bits))),
        #                  S1, S2, S1 @ S2 ]
        self.group = generate_freegroup(swap_ele)


        @qml.qnode(device=self.dev, diff_method="backprop", interface="torch")
        def observable_circ():
            qml.PauliZ(num_bits-1)
            return qml.expval(qml.Identity(wires=range(num_bits)))
        self.observable = qml.matrix(observable_circ)()
        

#%%
if __name__ == "__main__":
    order = 2
    num_bits = 6
    symmetry = SwapSymmetry(num_bits)

    num_data = 16
    train_data = encoded_data_set(num_data)
    test_data = encoded_data_set(num_data*6)
    
    with open('data_symmetry5.pkl', 'wb') as f:
        pickle.dump([num_bits, train_data, test_data, symmetry], f)


