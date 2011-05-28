
from nose.tools import eq_

from libctw import ctw


def test_children_respecting():
    model = ctw.create_model(deterministic=True)
    model.see_generated("01")
    model.switch_history()
    model.see_generated("0")

    p_estim = 0
    p_uncovered = 0.5 * 0.5
    p_child = 0.5
    eq_(model.root.pw, 0.5 * (p_estim + p_child * 1.0 * p_uncovered))

