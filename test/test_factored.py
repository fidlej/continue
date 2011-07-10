
from nose.tools import eq_

from libctw import factored

def test_offset():
    model = factored.create_model(num_factors=3)
    eq_(model.offset, 0)
    model.see_generated([1, 0, 0])
    eq_(model.offset, 0)
    model.see_added([1, 0])
    eq_(model.offset, 0)
    model.see_generated([1])
    eq_(model.offset, 1)
    model.see_generated([0])
    eq_(model.offset, 2)
    model.see_generated([0])
    eq_(model.offset, 0)


def test_revert_generated():
    model = factored.create_model(deterministic=True, max_depth=2,
            num_factors=3)
    model.see_generated([1, 0, 0])
    model.see_generated([1, 0, 0])
    eq_(model.predict_one(), 1.0)

    model.see_generated([1])
    eq_(model.predict_one(), 0.0)
    model.revert_generated(1)
    eq_(model.predict_one(), 1.0)

    model.revert_generated(6)
    eq_(model.offset, 0)
    eq_(model.predict_one(), 0.5)
