
from nose.tools import eq_

from libctw import ctw

from test_ctw import eq_float_


def test_children_respecting():
    model = ctw.create_model(deterministic=True)
    model.see_generated([0, 1])
    model.switch_history()
    model.see_generated([0])

    p_estim = 0
    p_uncovered = 0.5 * 0.5
    p_child = 0.5
    eq_(model.get_history_p(), 0.5 * (p_estim + p_child * 1.0 * p_uncovered))


def test_predict_from_zero_history():
    model = ctw.create_model(deterministic=True)
    model.see_generated([1, 1])
    model.switch_history()

    p_estim = 0.5
    p_uncovered = 0.5
    p_child = 0.5
    pw = 0.5 * (p_estim + p_uncovered * p_child * 1.0)
    eq_float_(model.get_history_p(), pw)

    p_estim = 0.5
    p_uncovered = 0.5 * 0.5
    p_child = 0.5
    pw = 0.5 * (p_estim + p_uncovered * p_child * 1.0)
    eq_float_(model.predict_one(), pw / model.get_history_p())
