
from nose.tools import eq_
import itertools

from libctw import ctw, naive_ctw


def test_kt_estim_update():
    _check_estim_update(ctw._kt_estim_update, naive_ctw._estim_kt_p)


def test_determ_estim_update():
    _check_estim_update(ctw._determ_estim_update, naive_ctw._estim_determ_p)


def test_empty_context():
    model = ctw.create_model()
    eq_(model.predict_one(), 0.5)


def test_see():
    contexted =naive_ctw._Contexted(naive_ctw._estim_kt_p)
    for seq in iter_all_seqs(seq_len=10):
        model = ctw.create_model()
        model.see(seq)
        eq_float_(model.root.pw, contexted.calc_p("", seq))


def test_predict_one():
    seq_len = 8
    for determ in [False, True]:
        for seq in iter_all_seqs(seq_len):
            model = ctw.create_model(determ)
            verifier = naive_ctw.create_model(determ)
            for c in seq:
                model.see(c)
                verifier.see(c)
                eq_float_(model.predict_one(), verifier.predict_one(),
                        precision=15)


def test_max_depth():
    model = ctw.create_model(max_depth=0)
    eq_(model.root.pw, 1.0)
    model.see("1")
    eq_(model.root.p_estim, 0.5)
    eq_(model.root.pw, model.root.p_estim)

    model.see("1")
    eq_(model.root.pw, naive_ctw._estim_kt_p(0, 2))


def test_max_depth_sum():
    for seq_len in xrange(10):
        total = 0.0
        for seq in iter_all_seqs(seq_len):
            model = ctw.create_model(max_depth=8)
            model.see(seq)
            total += model.root.pw

        eq_(total, 1.0)


def test_max_depth_example():
    # The calculated probablities are from the
    # 'Reflections on "The Context-Tree Weighting Method: Basic Properties"'
    # paper (figure 6 and 7).
    model = ctw.create_model(max_depth=3, past="110")
    model.see("0100110")
    p_seq = model.root.pw
    eq_float_(p_seq, 7/2048.0)

    model.see("0")
    p_seq2 = model.root.pw
    eq_float_(p_seq2, 153/65536.0)



def iter_all_seqs(seq_len):
    for seq in itertools.product(["0", "1"], repeat=seq_len):
        yield "".join(seq)


def _check_estim_update(estim_update, estimator):
    for seq_len in xrange(10):
        for bits in itertools.product([0, 1], repeat=seq_len):
            p = 1.0
            counts = [0, 0]
            for bit in bits:
                p *= estim_update(bit, counts)
                counts[bit] += 1

            eq_float_(p, estimator(*counts))


def eq_float_(value, expected, precision=16):
    eq_("%.*f" % (precision, value),
            "%.*f" % (precision, expected))
