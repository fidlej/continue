
from libctw import byting, formatting


def advance(model):
    """Generates deterministically the next bit.
    Returns a (bit, prediction_probability) pair.
    """
    one_p = model.predict_one()
    assert 0 <= one_p <= 1.0, "invalid P: %s" % one_p

    if one_p >= 0.5:
        bit, p = 1, one_p
    else:
        bit, p = 0, (1 - one_p)

    model.see_generated([bit])
    symbol = "1" if bit else "0"
    return symbol, p


class AllGenerated:
    def is_generated(self, bit_number):
        return True


class Interlaced:
    """Tells that the added and generated bits are interlaced:
        bits = added + generated + added + generated + ...
    """
    def __init__(self, num_added_bits, num_generated_bits):
        self.num_added_bits = num_added_bits
        self.num_generated_bits = num_generated_bits

    def is_generated(self, bit_number):
        index = bit_number % (self.num_added_bits + self.num_generated_bits)
        return index >= self.num_added_bits


def train_model(model, train_seqs, bytes=False, source_info=AllGenerated()):
    for seq in train_seqs:
        print "seq:", seq
        if bytes:
            seq = byting.to_binseq(seq)

        training_bits = formatting.to_bits(seq)
        for i, bit in enumerate(training_bits):
            if source_info.is_generated(i):
                model.see_generated([bit])
            else:
                model.see_added([bit])

        model.switch_history()


