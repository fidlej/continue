"""An implementation of efficient CTW model updating.

The code is inspired by the CTW implementation for MC-AIXI:
"A Monte-Carlo AIXI Approximation"
Joel Veness, Kee Siong Ng, Marcus Hutter, William Uther, David Silver
http://jveness.info/software/default.html
"""

def create_model(deterministic=False, max_depth=None, past=""):
    if deterministic:
        estim_update = _determ_estim_update
    else:
        estim_update = _kt_estim_update

    return _CtModel(estim_update, max_depth, past)


NO_CHILDREN = [None, None]


class _CtModel:
    def __init__(self, estim_update, max_depth=None, past=""):
        """Creates a Context Tree model.

        The given past is outside of the model scope.
        Its P(past) is not bound to this model.
        """
        self.estim_update = estim_update
        self.max_depth = max_depth
        self.history = []
        self.past_len = len(past)
        self.switch_number = 0
        self.see_added([_to_bit(c) for c in past])

        # Now the self is ready for use.
        self.root = _Node()

    def see_generated(self, seq):
        for c in seq:
            bit = _to_bit(c)
            self._see_generated_bit(bit)

    def switch_history(self):
        """Keeps the learned model, but starts
        with an empty history.
        It allows to switch between sequence examples.
        """
        self.switch_number += 1
        self.history = []

    def see_added(self, bits):
        """Adds a historic bit without affecting the model parameters.
        The bit is outside of the model scope.
        Its P(bit) is not bound to this model.
        """
        # Note that it is enough to keep just the first and the last
        # max_depth bits of history.
        self.history += bits

    def _see_generated_bit(self, bit):
        """Updates the counts and precomputed probabilities
        on the context path.
        """
        context = self._get_context()
        path = _get_context_path(self.root, context, save_nodes=True)

        for i, node in enumerate(reversed(path)):
            node.p_estim *= self.estim_update(bit, node.counts)
            node.counts[bit] += 1

            # No weighting is used, if the node has no children.
            if node.children == NO_CHILDREN:
                node.pw = node.p_estim
            else:
                if node.seen_switch != self.switch_number:
                    node_context = context[i:]
                    self._update_after_switch(node, node_context)

                p0context = _child_pw(node, 0)
                p1context = _child_pw(node, 1)
                node.pw = 0.5 * (node.p_estim +
                    p0context * p1context * node.p_uncovered)

        self.history.append(bit)

    def predict_one(self):
        """Computes the conditional probability
        P(Next_bit=1|history).
        """
        if self.root.pw == 0:
            raise ValueError(
                    "Impossible history. Try non-deterministic prior.")

        bit = 1
        context = self._get_context()
        path = _get_context_path(self.root, context)
        assert len(path) == len(context) + 1

        new_pw = None
        for i, (child_bit, node) in enumerate(
                zip([None] + context, reversed(path))):
            p_estim = node.p_estim * self.estim_update(bit, node.counts)
            # The NO_CHILDREN test would not be enough
            # if save_nodes=False is used.
            if new_pw is None and node.children == NO_CHILDREN:
                new_pw = p_estim
            else:
                p_uncovered = node.p_uncovered
                if node.seen_switch != self.switch_number:
                    node_context = context[i:]
                    p_uncovered *= self._get_p_uncovered(node_context)

                # The context can be shorter than
                # the existing tree depth, when switching history.
                # Both children will carry valid probability.
                if child_bit is None:
                    child_bit = 0
                    new_pw = _child_pw(node, child_bit)

                p_other_child_pw = _child_pw(node, 1 - child_bit)
                new_pw = 0.5 * (p_estim +
                    new_pw * p_other_child_pw * p_uncovered)

        return new_pw / float(self.root.pw)

    def _get_p_uncovered(self, subcontext):
        """Returns the probability of the bit uncovered
        by the subcontext children.

        The "The context-tree weighting method: Extensions" paper
        uses "has a tail" wording for subcontexts with an uncovered bit.
        """
        if self.max_depth is not None and len(subcontext) == self.max_depth:
            return 1.0

        if len(subcontext) < self.past_len:
            return 1.0

        # A bit is uncovered, if the history
        # starts with the subcontext.
        if self.history[:len(subcontext)] == subcontext:
            return 0.5
        return 1.0

    def _get_context(self):
        """Returns the recent context.
        """
        context = self.history
        if self.max_depth is not None and len(context) > self.max_depth:
            context = context[len(context) - self.max_depth:]
            assert len(context) == self.max_depth
        return context

    def _update_after_switch(self, node, subcontext):
        node.p_uncovered *= self._get_p_uncovered(subcontext)
        node.seen_switch = self.switch_number


def _get_context_path(root, context, save_nodes=False):
    """Returns a path from the root to the start of the context.
    """
    path = [root]
    node = root
    for i, bit in enumerate(reversed(context)):
        child = node.children[bit]
        if child is None:
            child = _Node()
            if save_nodes:
                node.children[bit] = child

        path.append(child)
        node = child

    return path


def _child_pw(node, child_bit):
    child = node.children[child_bit]
    if child is None:
        return 1.0
    return child.pw


def _to_bit(symbol):
    return "01".index(symbol)


class _Node:
    def __init__(self):
        self.p_estim = 1.0
        self.pw = 1.0
        self.p_uncovered = 1.0
        self.seen_switch = None
        self.counts = [0, 0]
        self.children = [None, None]

    def __repr__(self):
        return "Node" + str(dict(
            p_estim=self.p_estim,
            pw=self.pw,
            p_uncovered=self.p_uncovered,
            counts=self.counts))


def _determ_estim_update(new_bit, counts):
    new_counts = counts[:]
    new_counts[new_bit] += 1
    if new_counts[0] > 0 and new_counts[1] > 0:
        return 0.0

    p_new = 0.0
    if new_counts[0] == 0:
        p_new += 0.5
    if new_counts[1] == 0:
        p_new += 0.5

    p_old = 0.0
    if counts[0] == 0:
        p_old += 0.5
    if counts[1] == 0:
        p_old += 0.5

    return p_new / float(p_old)


def _kt_estim_update(new_bit, counts):
    """Computes P(Next_bit=new_bit|counts)
    for the the Krichevski-Trofimov estimator.
    """
    return (counts[new_bit] + 0.5) / float((sum(counts) + 1))

