
class SuffixExtractor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def extract_context(self, history):
        """Extracts the history suffix.
        """
        context = history
        if self.max_depth is not None and len(context) > self.max_depth:
            context = context[len(context) - self.max_depth:]
            assert len(context) == self.max_depth
        return context


class VarExtractor:
    """An extractor of custom contexts.
    A context is extracted from different places in the history.
    The values of the specified variables form the context.
    """
    def __init__(self, root_var, max_depth=None):
        self.root_var = root_var
        self.suffix_extractor = SuffixExtractor(max_depth)
        #TODO: assert that the depth of the vars isn't bigger that max_depth

    def extract_context(self, history):
        """Extracts a context based on the tree of var indexes.
        After extracting all the var values, suffixes are used as a fallback.
        """
        context = []
        used_indexes = []

        var = self.root_var
        while var is not None:
            if -var.index > len(history):
                break

            used_indexes.append(var.index)
            bit = history[var.index]
            context.insert(0, bit)
            var = var.children[bit]

        return self._get_unused_suffix(history, used_indexes) + context

    def _get_unused_suffix(self, history, used_indexes):
        context = self.suffix_extractor.extract_context(history)
        if used_indexes:
            context = context[:]
            used_indexes.sort()
            for index in used_indexes:
                if -index < len(context):
                    del context[index]
                else:
                    del context[0]
        return context


class Var:
    """A var specifies what to prepend to the context.
    The var index is the index of a bit in the history.
    The index is negative (e.g., -1 for the last bit).
    The corresponding children[bit_value] var is consulted next.
    """
    def __init__(self, index, children=(None, None)):
        assert index < 0
        self.index = index
        self.children = children

