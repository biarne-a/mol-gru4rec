import torch


class RetrievalMetrics:
    def __init__(self, at_k_list: list[int]):
        self.at_k_list = at_k_list
        self.top_k_ids = []
        self.target_ids = []

    def update(self, top_k_ids: torch.Tensor, target_ids: torch.Tensor):
        self.top_k_ids.append(top_k_ids)
        self.target_ids.append(target_ids)

    def compute(self):
        top_k_ids = torch.cat(self.top_k_ids, dim=0)
        target_ids = torch.cat(self.target_ids, dim=0)

        _, rank_indices = torch.max(
            torch.cat([top_k_ids, target_ids], dim=1) == target_ids,
            dim=1,
        )
        ranks = rank_indices + 1
        output = {}
        for at_k in self.at_k_list:
            output[f"hr@{at_k}"] = (ranks <= at_k).to(torch.float32).mean().item()
        output["mrr"] = (1.0 / ranks).mean().item()

        return output
