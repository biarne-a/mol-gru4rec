import abc

import torch


class SimilarityModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pass
