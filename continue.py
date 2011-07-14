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

from libctw import modeling, byting, formatting
from libctw.anycontext import creating

DEFAULTS = {
        "num_predicted_bits": 100,
        "estimator": "kt",
        "train": [],
        }

def _parse_args():
    parser = optparse.OptionParser(__doc__)
    parser.add_option("-n", dest="num_predicted_bits", type="int",
            help="the number of bits to predict (default=%(num_predicted_bits)s)" % DEFAULTS)
    parser.add_option("-d", "--depth", type="int",
            help="limit max depth of the suffix tree model")
    parser.add_option("-e", "--estimator", choices=["kt", "determ"],
            help="use Krichevski-Trofimov or deterministic prior [kt|determ] (default=%(estimator)s)" % DEFAULTS)
    parser.add_option("-b", "--bytes", action="store_true",
            help="accept and predict a sequence of bytes")
    parser.add_option("-t", "--train", action="append",
            help="train the model on the given training sequence")
    parser.add_option("-f", "--file",
            help="take training sequences from the file")
    parser.add_option("-g", "--gain", action="store_true",
            help="use information gain for context selection")
    parser.set_defaults(**DEFAULTS)

    options, args = parser.parse_args()
    if len(args) != 1:
        parser.error("A sequence is expected.")

    seq = args[0]
    if not options.bytes and len(seq.strip("01")) > 0:
        parser.error("Expecting a sequence of 0s and 1s.")

    if options.file:
        with open(options.file) as input:
            options.train += [line.rstrip("\n\r") for line in
                    input.readlines()]

    return options, seq


def _format_products(parts):
    return " * ".join("%.2f" % p for p in parts)


def _round_up(value, base):
    whole = value - (value % base)
    if whole < value:
        whole += base
    return whole


def _create_history(options, input_seq):
    if options.bytes:
        seq = byting.to_binseq(input_seq)
    else:
        seq = input_seq
    return formatting.to_bits(seq)


def _create_model(options, history):
    deterministic = options.estimator == "determ"
    if options.bytes:
        factored = True
        historian = creating.Historian(history, 8, 0)
    else:
        factored = False
        historian = creating.Historian(history, 1, 0)

    if options.gain:
        min_var_index = None
    else:
        min_var_index = creating.SUFFIXES_ONLY

    model = creating.create_model(historian,
            factored=factored,
            deterministic=deterministic,
            max_depth=options.depth,
            min_var_index=min_var_index)

    if options.train:
        modeling.train_model(model, options.train, options.bytes)

    model.see_generated(history)
    return model


def main():
    options, input_seq = _parse_args()
    history = _create_history(options, input_seq)
    model = _create_model(options, history)

    if options.bytes:
        num_predicted_bits = _round_up(options.num_predicted_bits, 8)
    else:
        num_predicted_bits = options.num_predicted_bits

    probs = []
    bits = ""
    probability = 1.0
    sys.stdout.write("%s -> " % formatting.to_seq(history))
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

    if len(probs) > 10:
        probs_info = _format_products(probs[:9]) + " * ..."
    else:
        probs_info = _format_products(probs)
    print "with P = %f = %s" % (probability, probs_info)


if __name__ == "__main__":
    main()
