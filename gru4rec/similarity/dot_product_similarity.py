import torch

from gru4rec.similarity.similarity_module import SimilarityModule


class DotProductSimilarity(SimilarityModule):
    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return torch.mm(query_embeddings, item_embeddings.t()), {}
