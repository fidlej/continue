
from nose.tools import eq_
import math

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
    eq_(model.get_history_log_p(),
            math.log(0.5 * (p_estim + p_child * 1.0 * p_uncovered)))


def test_predict_from_zero_history():
    model = ctw.create_model(deterministic=True)
    model.see_generated([1, 1])
    model.switch_history()

    p_estim = 0.5
    p_uncovered = 0.5
    p_child = 0.5
    pw = 0.5 * (p_estim + p_uncovered * p_child * 1.0)
    eq_float_(model.get_history_log_p(), math.log(pw))

    p_estim = 0.5
    p_uncovered = 0.5 * 0.5
    p_child = 0.5
    pw = 0.5 * (p_estim + p_uncovered * p_child * 1.0)
    eq_float_(model.predict_one(), pw / math.exp(model.get_history_log_p()))


def test_invalid_history():
    model = ctw.create_model(deterministic=True)
    model.see_generated([0, 1, 1])
    model.switch_history()

    model.see_generated([0, 1])
    try:
        model.see_generated([0])
        assert False, "ValueError is expected"
    except ValueError, e:
        eq_(str(e), "Impossible history. Try non-deterministic prior.")

