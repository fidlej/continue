
from nose.tools import eq_
import itertools

from libctw import ctw

def test_estim_kt_p():
    # The tabulated values are from "Reflections on CTW".
    eq_(ctw._estim_kt_p(0, 1), 0.5)
    eq_(ctw._estim_kt_p(1, 0), 0.5)
    eq_(ctw._estim_kt_p(0, 2), 3/8.0)
    eq_(ctw._estim_kt_p(0, 5), 63/256.0)

    eq_(ctw._estim_kt_p(3, 1), 5/128.0)
    eq_(ctw._estim_kt_p(3, 5), 45/32768.0)

    eq_(ctw._estim_kt_p(0, 0), 1.0)


def test_estim_kt_p_sum():
    def p_func(seq):
        counts = ctw._count_followers("", seq)
        return ctw._estim_kt_p(*counts)

    _check_p_sum(p_func)


def test_estim_p_sum():
    def p_func(seq):
        counts = ctw._count_followers("", seq)
        return ctw._estim_p(*counts)

    _check_p_sum(p_func)


def test_count_followers():
    eq_(ctw._count_followers("", ""), (0, 0))
    eq_(ctw._count_followers("", "00111"), (2, 3))
    eq_(ctw._count_followers("0", "00111"), (1, 1))
    eq_(ctw._count_followers("1", "00111"), (0, 2))
    eq_(ctw._count_followers("010", "00101011"), (0, 2))

    eq_(ctw._count_followers("", "0"), (1, 0))
    eq_(ctw._count_followers("", "1"), (0, 1))
    eq_(ctw._count_followers("0", "0"), (0, 0))
    eq_(ctw._count_followers("1", "1"), (0, 0))


def test_calc_p():
    eq_(ctw._calc_p("", ""), 1.0)
    eq_(ctw._calc_p("", "0"), 0.5)
    eq_(ctw._calc_p("", "1"), 0.5)
    eq_(ctw._calc_p("", "01"), ctw._calc_p("", "10"))
    eq_(ctw._calc_p("", "11"), ctw._calc_p("", "00"))


def test_calc_p_sum():
    def p_func(seq):
        return ctw._calc_p("", seq)

    _check_p_sum(p_func)


def _check_p_sum(p_func):
    for seq_len in xrange(10):
        total = 0.0
        parts = []
        for bits in itertools.product("01", repeat=seq_len):
            seq = "".join(bits)
            seq_p = p_func(seq)
            total += seq_p
            parts.append((seq, seq_p))

        eq_(total, 1.0, "%s != 1.0, probabilities: %s" % (total, parts))

