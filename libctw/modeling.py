

def advance(model):
    """Generates deterministically the next bit.
    Returns a (bit, prediction_probability) pair.
    """
    one_p = model.predict_one()
    assert 0 <= one_p <= 1.0

    if one_p >= 0.5:
        bit, p = "1", one_p
    else:
        bit, p = "0", (1 - one_p)

    model.see(bit)
    return bit, p

