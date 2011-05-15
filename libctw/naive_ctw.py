"""
A Context-Tree Weighting model.

Resource:
1) The Context-Tree Weighting Method: Basic Properties
2) Reflections on "The Context-Tree Weighting Method: Basic Properties"
http://www.sps.ele.tue.nl/members/F.M.J.Willems/RESEARCH_files/CTW/ResearchCTW.htm
"""

import math


def create_model(deterministic=False):
    if deterministic:
        estimator = _estim_determ_p
    else:
        estimator = _estim_kt_p

    return _CtwModel(estimator)


class _CtwModel:
    def __init__(self, estimator):
        self.contexted = _Contexted(estimator)
        self.seen = ""

    def see(self, seq):
        #TODO: use an algorithm without the need
        # to store the seen sequence.
        # Only the counts are important for a context-based model.
        self.seen += seq

    def predict_one(self):
        """Computes the conditional probability
        P(next_bit=1|seen_bits).
        """
        #TODO: reuse previous computations
        return (self.contexted.calc_p("", self.seen + "1") /
                float(self.contexted.calc_p("", self.seen)))


class _Contexted:
    def __init__(self, estimator):
        self.estimator = estimator

    def calc_p(self, context, seq):
        """Estimates the probability of some bits from the given sequence.
        Only the bits following the context
        are considered by the probability.
        """
        num_zeros, num_ones = _count_followers(context, seq)
        if num_zeros == 0 and num_ones == 0:
            return 1.0

        p0context = self.calc_p("0" + context, seq)
        p1context = self.calc_p("1" + context, seq)
        p_uncovered = 1.0
        if seq.startswith(context):
            # A bit will be uncovered by the child models,
            # if the 0context and 1context don't fit before it in the sequence.
            # The "Extending the Context-Tree Weighting Method" paper names
            # the p_uncovered as P^{epsilon s}.
            assert self.estimator(1, 0) == self.estimator(0, 1)
            p_uncovered = 0.5

        # The CTW estimate is the average
        # of the this context model and the model of its children.
        # The recursive averaging prefers simpler models.
        result = 0.5 * (
                self.estimator(num_zeros, num_ones) +
                p0context * p1context * p_uncovered)
        return result


def _count_followers(context, seq):
    # Efficiency is ignored here.
    # Better algorithms exist.
    num_zeros = 0
    num_ones = 0
    for i in xrange(len(seq) - len(context)):
        if seq[i:].startswith(context):
            if seq[i + len(context)] == "0":
                num_zeros += 1
            else:
                num_ones += 1

    return num_zeros, num_ones


def _estim_determ_p(num_zeros, num_ones):
    """An estimator that beliefs just in
    deterministic memory-less models.
    """
    p = 0.0
    if num_zeros == 0:
        p += 0.5
    if num_ones == 0:
        p += 0.5
    return p


def _estim_kt_p(num_zeros, num_ones):
    """Estimates the probability of the given numbers.
    Assumes a memory-less model:
        P(num_zeros, num_ones) = theta**num_ones * (1 - theta)**num_zeros

        with Dirichlet(1/2.0,1/2.0) prior P(theta).
    The resulting Bayesian mixture is a "Krichevski-Trofimov" estimator.
    """
    a_mul = 1.0
    for i in xrange(num_zeros):
        a_mul *= i + 0.5

    b_mul = 1.0
    for i in xrange(num_ones):
        b_mul *= i + 0.5

    return a_mul * b_mul / float(math.factorial(num_zeros + num_ones))

