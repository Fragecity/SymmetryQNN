class MidCircuit:
    """Base class for circuits in quantum neural networks"""
    def __init__(self):
        self.name = ""
        pass

    def circuit(self, params):
        pass

    def num_params(self):
        pass
