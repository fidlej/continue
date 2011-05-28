
from libctw import ctw

def create_model(deterministic=False, max_depth=None, num_factors=8):
    cts = []
    for i in xrange(num_factors):
        cts.append(ctw.create_model(deterministic, max_depth))

    return _Factored(cts)


class _Factored:
    def __init__(self, cts):
        self.cts = cts
        self.offset = 0

    def see_generated(self, seq):
        for c in seq:
            for i, ct in enumerate(self.cts):
                if i == self.offset:
                    ct.see_generated(c)
                else:
                    bit = 1 if c == "1" else 0
                    ct.see_added([bit])

            self.offset = (self.offset + 1) % len(self.cts)

    def predict_one(self):
        return self.cts[self.offset].predict_one()

    def switch_history(self):
        self.offset = 0
        for ct in self.cts:
            ct.switch_history()
