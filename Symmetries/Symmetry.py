from Utils import dagger, Adjoint, hs_norm

class Symmetry():
    '''
    Symmetry Interface
    Ps: deployed symmetry group and the global observable
    Ms: exact methods for cc symmetry guidance for given specific symmetry and ob.
    '''
    def __init__(self) -> None:
        self.group = []
        self.observable = 0

    def _twirling(self, O_tilde):
        '''twirling O_tilde'''
        SOS_list = [Adjoint(S, O_tilde) for S in self.group ]
        return 1/len(SOS_list) * sum(SOS_list)

    def _get_O_PO(self, U):
        '''return [O_tilde = U*self.ob*U^dag, twirled O_tide] for the given U'''
        O_tilde = Adjoint(U, self.observable)
        PO = self._twirling(O_tilde)
        return O_tilde, PO
    
    def symmetry_guidance(self, U):
        '''the regularization term value for the given U'''
        O_tilde, PO = self._get_O_PO(U)
        return hs_norm(O_tilde - PO) / len(U)