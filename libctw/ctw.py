

def create_model(deterministic=False):
    if deterministic:
        estim_update = _determ_estim_update
    else:
        estim_update = _kt_estim_update

    return _CtModel(estim_update)


class _CtModel:
    def __init__(self, estim_update):
        self.estim_update = estim_update
        self.history = []
        self.root = _Node()

    def see(self, seq):
        for c in seq:
            bit = 1 if c == "1" else 0
            self._see_bit(bit)

    def _see_bit(self, bit):
        """Updates the counts and precomputed probabilities
        on the context path.
        """
        context = self.history
        path = _get_context_path(self.root, context, save_nodes=True)

        for step, node in enumerate(reversed(path)):
            p0context = _child_pw(node, 0)
            p1context = _child_pw(node, 1)
            p_uncovered = self._get_p_uncovered(context[step:])

            node.p_estim *= self.estim_update(bit, node.counts)
            node.pw = 0.5 * (node.p_estim +
                p0context * p1context * p_uncovered)
            node.counts[bit] += 1

        self.history.append(bit)

    def predict_one(self):
        """Computes the conditional probability
        P(Next_bit=1|history).
        """
        bit = 1
        context = self.history
        path = _get_context_path(self.root, context)

        child_pw = 1.0
        for step, (child_bit, node) in enumerate(
                zip(reversed(context + [None]), reversed(path))):
            p_estim = node.p_estim * self.estim_update(bit, node.counts)
            if child_bit is None:
                child_pw = p_estim
            else:
                p_other_child_pw = _child_pw(node, 1 - child_bit)

                p_uncovered = self._get_p_uncovered(context[step:])
                child_pw = 0.5 * (p_estim +
                    child_pw * p_other_child_pw * p_uncovered)

        return child_pw / float(self.root.pw)

    def _get_p_uncovered(self, subcontext):
        """Returns the probability of the bit uncovered
        by the subcontext children.
        """
        if self.history[:len(subcontext)] == subcontext:
            return 0.5
        return 1.0


def _child_pw(node, child_bit):
    child = node.children[child_bit]
    if child is None:
        return 1.0
    return child.pw


def _get_context_path(root, context, save_nodes=False):
    path = [root]
    node = root
    for bit in reversed(context):
        child = node.children[bit]
        if child is None:
            child = _Node()
            if save_nodes:
                node.children[bit] = child
        path.append(child)
        node = child

    return path


class _Node:
    def __init__(self):
        self.p_estim = 1.0
        self.pw = 1.0
        self.counts = [0, 0]
        self.children = [None, None]

    def __repr__(self):
        return "Node" + str(dict(
            p_estim=self.p_estim,
            pw=self.pw,
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

