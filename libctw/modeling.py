
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


def train_model(model, train_seqs, bytes=False):
    for i, seq in enumerate(train_seqs):
        if bytes:
            seq = byting.to_binseq(seq)

        training_bits = formatting.to_bits(seq)
        model.see_generated(training_bits)
        model.switch_history()


