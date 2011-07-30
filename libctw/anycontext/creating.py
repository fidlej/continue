
from libctw import ctw, extracting
from libctw import factored as _factored
from libctw.anycontext import selecting

SUFFIXES_ONLY = 0

def create_model(historian, factored=False, deterministic=False,
        max_depth=None, min_var_index=None):
    if factored:
        factors = []
        for positions in historian.get_factored_positions():
            root_var = selecting.select_vartree(
                    historian.get_history(),
                    positions,
                    min_index=min_var_index)
            factors.append(_create_factor(root_var, deterministic, max_depth))

        return _factored.create_factored_model(factors)
    else:
        positions = historian.get_generated_positions()
        root_var = selecting.select_vartree(
                historian.get_history(),
                positions,
                min_index=min_var_index)
        return _create_factor(root_var, deterministic, max_depth)


def _create_factor(root_var, deterministic, max_depth):
    extractor = extracting.VarExtractor(root_var, max_depth)
    return ctw.create_context_based_model(extractor,
            deterministic=deterministic)


class Historian:
    """It assumes that the history consists of many steps.
    Each step has a fixed number of generated bits,
    followed by a fixed number of added bits.
    """
    def __init__(self, history, num_generated_bits, num_added_bits):
        assert len(history) % (num_generated_bits + num_added_bits) == 0
        self.history = history
        self.num_generated_bits = num_generated_bits
        self.num_added_bits = num_added_bits

    def get_history(self):
        return self.history

    def get_generated_positions(self):
        positions = []
        for start in self._get_step_starts():
            for i in xrange(self.num_generated_bits):
                positions.append(start + i)
        return positions

    def get_factored_positions(self):
        on_factors = []
        for i in xrange(self.num_generated_bits):
            positions = []
            for start in self._get_step_starts(offset=i):
                positions.append(start)

            on_factors.append(positions)
        return on_factors

    def get_steps(self):
        """Returns (generated, added) bits for each step.
        """
        steps = []
        for start in self._get_step_starts():
            generated_end = start + self.num_generated_bits
            added_start = generated_end
            added_end = added_start + self.num_added_bits
            generated = self.history[start:generated_end]
            added = self.history[added_start:added_end]
            steps.append((generated, added))

        return steps

    def _get_step_starts(self, offset=0):
        step_len = self.num_generated_bits + self.num_added_bits
        num_steps = len(self.history) // step_len
        return [offset + i * step_len for i in xrange(num_steps)]



