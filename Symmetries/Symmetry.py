from Utils import Adjoint, hs_norm


class Symmetry:
    '''
    Symmetry Interface

    Properties: 
    
    self.group (List): deployed symmetry freeGroup
    self.observable : the global observable

    Methods:

    self._twirling(O_tilde): twirling the given O_tilde (unrelated to self.ob) using #!self.group
    |
    V
    self._get_O_PO(U): return [O_tilde = U*O*U^dag] for the given U using #!self.ob and self._twirling
    since U of the circuit does not contain the observable O
    |
    V
    self.symmetry_guidance(U): return the regularization term value for the given U using #!self.ob and above

    Args

    U/O_tilde: a matrix or a sympy symboled matrix
    '''

    def __init__(self) -> None:
        self.group = []
        self.observable = 1

    def _twirling(self, O_tilde):
        '''twirling given O_tilde (unrelated to self.ob) using self.group'''
        SOS_list = [Adjoint(S, O_tilde) for S in self.group]
        return 1 / len(SOS_list) * sum(SOS_list)

    def _get_O_PO(self, U):
        '''return [O_tilde = U*self.ob*U^dag, twirled O_tide] for the given U'''
        O_tilde = Adjoint(U, self.observable)
        PO = self._twirling(O_tilde)
        return O_tilde, PO

    def symmetry_guidance(self, U):
        '''the regularization term value for the given U'''
        O_tilde, PO = self._get_O_PO(U)
        return hs_norm(O_tilde - PO) / len(U)
