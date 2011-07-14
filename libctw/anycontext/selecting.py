
import math
import logging

from libctw import extracting

def select_vartree(history, positions, min_index=None, max_depth=None):
    """Returns a tree of vars.
    The selected vars should be helpful in classifying
    the bits on the given positions.
    """
    if not positions:
        return None

    if min_index is None:
        min_index = -len(history)
    else:
        min_index = max(-len(history), min_index)

    var_indexes = range(-1, min_index - 1, -1)
    logging.info("building tree for factor %s", positions[0])
    return _TreeBuilder(history, max_depth).build_tree(
            positions, var_indexes)


class _TreeBuilder:
    def __init__(self, history, max_depth):
        self.history = history
        self.max_depth = max_depth

    def build_tree(self, positions, var_indexes, depth=0):
        if self.max_depth is not None and depth > self.max_depth:
            return None

        var_index, on_0, on_1 = _choose_split(self.history, positions,
                var_indexes)
        logging.info("best var at %s: %s", depth, var_index)
        if var_index is None:
            return None

        unused_indexes = var_indexes[:]
        unused_indexes.remove(var_index)
        next_depth = depth + 1
        children = (
                self.build_tree(on_0, unused_indexes, next_depth),
                self.build_tree(on_1, unused_indexes, next_depth))
        return extracting.Var(var_index, children)


def _choose_split(history, positions, var_indexes):
    """Select the best predictor of the bit on the given positions.
    The index of the predictor is an index in the history (e.g., -1).
    Returns (var_index, positions_with_0_in_var, positions_with_1_in_var).
    """
    best_var_index = None
    best_on_0 = None
    best_on_1 = None
    min_complexity = None
    for var_index in var_indexes:
        on_0 = _filter_poss(history, positions, var_index, 0)
        on_1 = _filter_poss(history, positions, var_index, 1)
        complexity = (_get_complexity(history, on_0) +
                _get_complexity(history, on_1))
        if min_complexity is None or complexity < min_complexity:
            min_complexity = complexity
            best_var_index = var_index
            best_on_0 = on_0
            best_on_1 = on_1

    return best_var_index, best_on_0, best_on_1


def _filter_poss(history, positions, var_index, needed_value):
    matching = []
    for pos in positions:
        abs_index = pos + var_index
        if abs_index < 0:
            # Missing vars are penalized by including them
            # to both branches.
            #TODO: What is the right way to handle missing vars?
            matching.append(pos)
        elif history[abs_index] == needed_value:
            matching.append(pos)

    return matching


def _get_complexity(history, positions):
    """Returns the expected number of bits
    needed to encode the sequence.
    """
    num_positive = _count_positive(history, positions)
    if num_positive == 0 or num_positive == len(positions):
        return 0.0

    p = num_positive / float(len(positions))
    return len(positions) * _entropy(p)


def _count_positive(history, positions):
    return sum(history[pos] for pos in positions)


def _entropy(p):
    return -(p * math.log(p, 2) + (1 - p) * math.log(1 - p, 2))


