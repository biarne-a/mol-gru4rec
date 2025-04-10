import torch
import torch.nn.functional as F

from gru4rec.similarity.similarity_module import SimilarityModule


class CosineSimilarity(SimilarityModule):
    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)
        return torch.mm(query_embeddings, item_embeddings.t()), {}
