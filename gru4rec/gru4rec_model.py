import torch
import torch.nn as nn

from gru4rec.metrics import RetrievalMetrics
from gru4rec.similarity.similarity_module import SimilarityModule
from train.data import Data


class Gru4RecModel(nn.Module):
    def __init__(
        self,
        data: Data,
        device: torch.tensor,
        similarity_module: SimilarityModule,
        aux_loss_weights: dict[str, float],
        embedding_dim: int,
        dropout_p_embed: float = 0.0,
        dropout_p_gru: float = 0.0,
    ):
        super(Gru4RecModel, self).__init__()
        all_item_ids = torch.tensor(data.movie_id_lookup.vocab_ids, device=device, dtype=torch.long)
        self._all_item_ids = all_item_ids.squeeze(0)
        self._vocab_size = data.movie_id_lookup.get_vocab_size()
        self._embedding_dim = embedding_dim
        self._movie_id_embedding = nn.Embedding(self._vocab_size, embedding_dim)
        self._dropout_emb = nn.Dropout(dropout_p_embed)
        self._gru_layer = nn.GRU(embedding_dim, embedding_dim, batch_first=True, dropout=dropout_p_gru)
        self._similarity_module = similarity_module
        self._aux_loss_weights = aux_loss_weights
        self._device = device
        self.metrics = RetrievalMetrics(at_k_list=[1, 10, 50, 200])
        self.max_top_k = self.metrics.at_k_list[-1]

    def _get_weighted_loss(
        self,
        main_loss: torch.Tensor,
        aux_losses: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        weighted_loss = main_loss
        for key, aux_loss in aux_losses.items():
            cur_weighted_loss = aux_loss * self._aux_loss_weights[key]
            weighted_loss = weighted_loss + cur_weighted_loss
        return weighted_loss

    def forward(self, inputs):
        ctx_movie_emb = self._movie_id_embedding(inputs["input_ids"])
        ctx_movie_emb = self._dropout_emb(ctx_movie_emb)
        gru_output, _ = self._gru_layer(ctx_movie_emb)
        sequence_lengths = torch.sum(inputs["input_mask"], dim=1).int()
        batch_size = inputs["input_ids"].size(0)
        batch_indexes = torch.arange(batch_size).to(self._device)
        last_hidden_state = gru_output[batch_indexes, sequence_lengths - 1, :]
        logits, aux_losses = self._similarity_module(
            query_embeddings=last_hidden_state,
            item_embeddings=self._movie_id_embedding.weight.unsqueeze(0),
            user_id=inputs["user_id"].squeeze(1),
        )
        return logits, aux_losses

    def train_step(self, inputs, optimizer, criterion):
        optimizer.zero_grad()
        y_true = inputs["label"]
        y_pred, aux_losses = self(inputs)
        main_loss = criterion(y_pred, y_true.squeeze())
        loss = self._get_weighted_loss(main_loss, aux_losses)
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_step(self, inputs, criterion):
        with torch.no_grad():
            y_true = inputs["label"]
            y_pred, aux_losses = self(inputs)
            main_loss = criterion(y_pred, y_true.squeeze())
            loss = self._get_weighted_loss(main_loss, aux_losses)
            _, top_k_indices = torch.topk(y_pred, dim=1, k=self.max_top_k)  # (B, k,)
            top_k_ids = self._all_item_ids[top_k_indices]
            self.metrics.update(top_k_ids, y_true)

        return loss.item()
