
from nose.tools import eq_
import itertools

from libctw import ctw, naive_ctw


def test_kt_estim_update():
    _check_estim_update(ctw._kt_estim_update, naive_ctw._estim_kt_p)


def test_determ_estim_update():
    _check_estim_update(ctw._determ_estim_update, naive_ctw._estim_determ_p)


def _check_estim_update(estim_update, estimator):
    precision = 16
    for seq_len in xrange(10):
        for bits in itertools.product([0, 1], repeat=seq_len):
            p = 1.0
            counts = [0, 0]
            for bit in bits:
                p *= estim_update(bit, counts)
                counts[bit] += 1

            eq_("%.*f" % (precision, p),
                    "%.*f" % (precision, estimator(*counts)))

