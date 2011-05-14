#!/usr/bin/env python
"""Usage: %prog BINARY_SEQUENCE
Continues the given binary sequence with the best guess.

Example:
$ continue.py 01101
01101 -> 1011011011
with P = 0.052825
"""

import sys
import optparse

from libctw import ctw, modeling, byting

DEFAULTS = {
        "num_predicted_bits": 10,
        }

def _parse_args():
    parser = optparse.OptionParser(__doc__)
    parser.add_option("-n", dest="num_predicted_bits", type="int",
            help="the number of bits to predict (default=%(num_predicted_bits)s)" % DEFAULTS)
    parser.add_option("-d", "--deterministic", action="store_true",
            help="assume deterministic sequence generator models")
    parser.add_option("-b", "--bytes", action="store_true",
            help="accept and predict a sequence of bytes")
    parser.set_defaults(**DEFAULTS)

    options, args = parser.parse_args()
    if len(args) != 1:
        parser.error("A sequence is expected.")

    seq = args[0]
    if not options.bytes and len(seq.strip("01")) > 0:
        parser.error("Expecting a sequence of 0s and 1s.")

    return options, seq


def main():
    options, input_seq = _parse_args()
    seq = input_seq
    num_predicted_bits = options.num_predicted_bits
    if options.bytes:
        seq = byting.to_bits(input_seq)
        num_predicted_bits *= 8

    model = ctw.create_model(options.deterministic)
    model.see(seq)

    probs = []
    bits = ""
    probability = 1.0
    sys.stdout.write("%s -> " % seq)
    for i in xrange(num_predicted_bits):
        bit, bit_p = modeling.advance(model)
        probs.append(bit_p)
        probability *= bit_p
        sys.stdout.write(str(bit))
        sys.stdout.flush()
        bits += str(bit)

    print
    if options.bytes:
        print "%s -> %s" % (input_seq, byting.to_bytes(bits))

    probs_info = " * ".join("%.2f" % p for p in probs)
    print "with P = %f = %s" % (probability, probs_info)


if __name__ == "__main__":
    main()
