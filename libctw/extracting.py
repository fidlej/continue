
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

