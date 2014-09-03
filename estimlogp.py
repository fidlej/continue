#!/usr/bin/env python
"""Usage: %prog PATH
Estimates the number of bits
to compress each line in the given file.
"""

import optparse
import math

from libctw import ctw

def _parse_args():
    parser = optparse.OptionParser(__doc__)
    parser.add_option("-d", "--depth", type="int",
            help="limit max depth of the suffix tree model")

    options, args = parser.parse_args()
    if len(args) != 1:
        parser.error("Expecting a path.")

    path = args[0]
    return options, path


def _tobit(c):
    if c == "1":
        return 1
    elif c == "0":
        return 0
    raise Exception("wrong bit: " + c)


def main():
    options, path = _parse_args()
    total_bits = 0
    n_seqs = 0
    for line in open(path):
        line = line.rstrip()
        bits = tuple(_tobit(c) for c in line)

        model = ctw.create_model(max_depth=options.depth)
        model.see_generated(bits)
        cost_nats = -model.get_history_log_p()
        cost_bits = cost_nats / math.log(2)
        total_bits += cost_bits
        n_seqs += 1
        print n_seqs, cost_bits

    print "avg cost: %s bits" % (total_bits / n_seqs)


if __name__ == "__main__":
    main()
