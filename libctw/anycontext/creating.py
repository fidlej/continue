
from libctw import ctw, extracting
from libctw.anycontext import selecting

SUFFIXES_ONLY = 0

def create_model(historian, factored=False, deterministic=False,
        max_depth=None, min_var_index=SUFFIXES_ONLY):
    assert factored == False
    #TODO: Support also a factored model.
    #TODO: Allow to specify select_vartree() limits.

    root_var = selecting.select_vartree(historian.get_history(),
            historian.get_generated_positions(),
            min_index=min_var_index)
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
        self.generated_shift = 0

    def get_history(self):
        return self.history

    def get_generated_positions(self):
        positions = []
        for start in self._get_step_starts():
            for i in xrange(self.num_generated_bits):
                positions.append(start + i)
        return positions

    def _get_step_starts(self, offset=0):
        step_len = self.num_generated_bits + self.num_added_bits
        num_steps = len(self.history) // step_len
        return [offset + i * step_len for i in xrange(num_steps)]



