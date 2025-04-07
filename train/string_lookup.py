import torch


class StringLookup:
    UNK_TOKEN = "<UNK>"

    def __init__(self, vocabulary):
        self.vocab = vocabulary
        self.keys_to_idx = {key: idx + 1 for idx, key in enumerate(vocabulary)}
        self.keys_to_idx[self.UNK_TOKEN] = 0
        self.idx_to_key = {idx: word for word, idx in self.keys_to_idx.items()}
        self.vocab_ids = list(range(len(self.keys_to_idx)))

    def lookup(self, keys) -> torch.Tensor:
        if isinstance(keys, list):
            return [self.keys_to_idx.get(str(key), 0) for key in keys]
        return self.keys_to_idx.get(keys, 0)

    def reverse_lookup(self, indices):
        if isinstance(indices, list):
            return [self.idx_to_key.get(i, self.UNK_TOKEN) for i in indices]
        return self.idx_to_key.get(indices, self.UNK_TOKEN)

    def get_vocab_size(self):
        return len(self.vocab_ids)
