
from nose.tools import eq_

from libctw.formatting import to_bits
from libctw.extracting import VarExtractor, Var

def test_var_following():
    extractor = VarExtractor(
            Var(-2, (
                Var(-1, (
                    None, Var(-4))),
                Var(-3)))
            )

    _assert_context(extractor, "000001", "000010")
    _assert_context(extractor, "000010", "000001")
    _assert_context(extractor, "110010", "110001")
    _assert_context(extractor, "001011", "001101")


def test_max_depth():
    extractor = VarExtractor(Var(-2), max_depth=3)
    _assert_context(extractor, "000001", "010")
    _assert_context(extractor, "000011", "011")
    _assert_context(extractor, "1", "1")
    _assert_context(extractor, "01", "10")
    _assert_context(extractor, "001", "010")

    extractor = VarExtractor(None, max_depth=0)
    _assert_context(extractor, "001", "")

    extractor = VarExtractor(None, max_depth=2)
    _assert_context(extractor, "001", "01")

    extractor = VarExtractor(Var(-2), max_depth=1)
    _assert_context(extractor, "01", "0")

    extractor = VarExtractor(Var(-4), max_depth=2)
    _assert_context(extractor, "001", "01")


def _assert_context(extractor, history_seq, expected_seq):
    eq_(extractor.extract_context(to_bits(history_seq)),
            to_bits(expected_seq))

