from numpy import trace
from Utils import hs_norm

#* These two are used for QNN in MyQC
def squareLoss(para, qnn):
    qnn.run_batch(para)
    # active = map(lambda, qnn_s.expects)
    return sum([( expectation - (-1)**label )**2   for expectation, label in zip(qnn.expects, qnn.labels)])

def accuracy(para, qnn):
    qnn.run_batch(para)
    loss_lst = [ 1 if (ep>0) == lab else 0 
                for ep, lab in zip(qnn.expects, qnn.labels) ]
    return 1 - sum(loss_lst)/ len(loss_lst)


def get_expects(O, rho):
    return trace(rho @ O).real

def cost(para, qnn, dataSet):
    O = qnn.get_O_tilde(para)
    csts = [(get_expects(O, rho) - label)**2 for rho, label in dataSet]
    return sum(csts)

def costG(para, qnnG, dataSet, symmetry, LAMBDA):
        O = qnnG.get_O_tilde(para)
        csts = [(get_expects(O, rho) - label)**2 for rho, label in dataSet]
        PO = symmetry._twirling(O)
        sg = hs_norm(PO - qnnG.O_tilde)
        return sum(csts) + LAMBDA * sg
    
def accuracy(para, qnn, dataSet):
    accy = 0
    O = qnn.get_O_tilde(para)
    for rho, label in dataSet:
        hyp = 1 if get_expects(O, rho)>=0 else -1
        if hyp - label < 1e-10: accy += 1
    return accy / len(dataSet)