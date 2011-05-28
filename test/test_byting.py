
from nose.tools import eq_

from libctw import byting


def test_to_bits():
    eq_(byting.to_binseq("A"), "01000001")
    eq_(byting.to_binseq("AC"), "0100000101000011")


def test_to_bytes():
    eq_(byting.to_bytes("01000001"), "A")
    eq_(byting.to_bytes("0100000101000011"), "AC")


def test_conversion():
    eq_(byting.to_bytes(byting.to_binseq("hello world")), "hello world")

