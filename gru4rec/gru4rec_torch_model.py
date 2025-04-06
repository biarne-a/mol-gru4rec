import torch
import torch.nn as nn
import torch.nn.functional as F


class Gru4RecModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dropout_p_embed=0.0, dropout_p_hidden=0.0):
        super(Gru4RecModel, self).__init__()
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._dropout_p_embed = dropout_p_embed
        self._dropout_p_hidden = dropout_p_hidden
        self._movie_id_embedding = nn.Embedding(vocab_size + 1, hidden_size)
        self._dropout_emb = nn.Dropout(dropout_p_embed)
        self._gru_layer = nn.GRU(hidden_size, hidden_size, batch_first=True, dropout=dropout_p_hidden)

    def forward(self, inputs):
        ctx_movie_emb = self._movie_id_embedding(inputs["input_ids"])
        ctx_movie_emb = self._dropout_emb(ctx_movie_emb)
        gru_output, _ = self._gru_layer(ctx_movie_emb)
        sequence_lengths = torch.sum(inputs["input_mask"], dim=1).int()
        batch_size = inputs["input_mask"].size(0)
        last_hidden_state_idx = torch.stack([torch.arange(batch_size), sequence_lengths - 1], dim=1)
        last_hidden_state = gru_output[last_hidden_state_idx[:, 0], last_hidden_state_idx[:, 1]]
        logits = torch.matmul(last_hidden_state, self._movie_id_embedding.weight.t())
        return logits

    def train_step(self, inputs, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        y_true = inputs["label"]
        y_pred = self(inputs)
        loss = criterion(y_pred, y_true.squeeze())
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_step(self, inputs, criterion):
        self.eval()
        with torch.no_grad():
            y_true = inputs["label"]
            y_pred = self(inputs)
            loss = criterion(y_pred, y_true.squeeze())
        return loss.item()
