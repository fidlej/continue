
from libctw import ctw

def create_model(deterministic=False, max_depth=None, num_factors=8):
    cts = []
    for i in xrange(num_factors):
        cts.append(ctw.create_model(deterministic, max_depth))

    return _Factored(cts)


def create_factored_model(bit_models):
    """Creates a factored model.
    Bits on different positions will be predicted
    by different models.
    """
    return _Factored(bit_models)


class _Factored:
    def __init__(self, cts):
        self.cts = cts
        self.offset = 0

    def see_generated(self, bits):
        for bit in bits:
            for i, ct in enumerate(self.cts):
                if i == self.offset:
                    ct.see_generated([bit])
                else:
                    ct.see_added([bit])

            self.offset = (self.offset + 1) % len(self.cts)

    def see_added(self, bits):
        for ct in self.cts:
            ct.see_added(bits)

    def predict_one(self):
        return self.cts[self.offset].predict_one()

    def switch_history(self):
        self.offset = 0
        for ct in self.cts:
            ct.switch_history()

    def revert_generated(self, num_bits):
        for ignored in xrange(num_bits):
            self.offset = (self.offset - 1) % len(self.cts)
            for i, ct in enumerate(self.cts):
                if i == self.offset:
                    ct.revert_generated(1)
                else:
                    ct.revert_added(1)
