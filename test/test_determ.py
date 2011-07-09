
from nose.tools import eq_

from libctw import ctw

def test_impossible_history():
    model = ctw.create_model(deterministic=True, max_depth=2)
    model.see_generated([0, 0, 0])
    eq_(0.0, model.predict_one())

    model.revert_generated(3)
    eq_(0.5, model.predict_one())
    model.see_generated([1, 1, 1])
    eq_(1.0, model.predict_one())

def test_sure():
    model = ctw.create_model(deterministic=True, max_depth=2)
    model.see_generated([1, 1, 1])
    eq_(1.0, model.predict_one())

