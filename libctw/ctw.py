

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
        #TODO: implement
        pass

    def predict_one(self):
        """Computes the conditional probability
        P(Next_bit=1|history).
        """
        context = self.history
        path = _get_context_path(self.root, context)

        child_pw = 1.0
        for step, (context_bit, node) in enumerate(
                zip(reversed(context), reversed(path))):
            p_estim = node.p_estim * self.estim_update(1, node.counts)
            other_child = node.childen[1 - context_bit]
            if other_child is None:
                p_other_child = 1.0
            else:
                p_other_child = other_child.pw

            p_uncovered = 1.0
            if self.history.strartswith(context[step:]):
                p_uncovered = 0.5

            child_pw = 0.5 * (p_estim +
                child_pw * p_other_child * p_uncovered)

        return child_pw


def _get_context_path(root, context):
    path = [root]
    node = root
    for bit in reversed(context):
        node = node.children[bit]
        if node is None:
            node = _Node()
        path.append(node)

    return path


class _Node:
    def __init__(self):
        self.p_estim = 1.0
        self.pw = 1.0
        self.counts = [0, 0]
        self.children = [None, None]


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
    return (counts[new_bit] + 0.5) / (sum(counts) + 1)

