class StringLookup:
    def __init__(self, vocabulary):
        self.vocab = vocabulary
        self.str_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.idx_to_str = {idx: word for word, idx in self.str_to_idx.items()}

    def lookup(self, strings):
        if isinstance(strings, list):
            return [self.str_to_idx.get(s, -1) for s in strings]
        return self.str_to_idx.get(strings, -1)

    def reverse_lookup(self, indices):
        if isinstance(indices, list):
            return [self.idx_to_str.get(i, "<UNK>") for i in indices]
        return self.idx_to_str.get(indices, "<UNK>")

    def __len__(self):
        return len(self.str_to_idx)
