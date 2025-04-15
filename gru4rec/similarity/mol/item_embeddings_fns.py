# Retrieval with Learned Similarities (RAILS, https://arxiv.org/abs/2407.15462).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Defines functions to generate item-side embeddings for MoL.

import abc
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from gru4rec.similarity.mol.embeddings_fn import MoLEmbeddingsFn, mask_mixing_weights_fn


def init_mlp_xavier_weights_zero_bias(m) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0.0)


class RecoMoLItemEmbeddingsFn(MoLEmbeddingsFn):
    """
    Generates P_X query-side embeddings for MoL based on input embeddings and other
    optional tensors for recommendation models. Tested for sequential retrieval
    scenarios.
    """

    def __init__(
        self,
        item_embedding_dim: int,
        item_dot_product_groups: int,
        item_semantic_embed: bool,
        item_semantic_emb_dimension: int,
        all_item_semantic_embeddings: torch.Tensor | None,
        dot_product_dimension: int,
        dot_product_l2_norm: bool,
        proj_fn: Callable[[int, int], torch.nn.Module],
        eps: float,
    ) -> None:
        super().__init__()

        self._item_emb_based_dot_product_groups: int = item_dot_product_groups - int(item_semantic_embed)
        self._item_emb_proj_module: torch.nn.Module = proj_fn(
            item_embedding_dim,
            dot_product_dimension * self._item_emb_based_dot_product_groups,
        )
        self._dot_product_dimension: int = dot_product_dimension
        self._dot_product_l2_norm: bool = dot_product_l2_norm
        self._item_semantic_embed = item_semantic_embed
        self._item_semantic_emb_dimension = item_semantic_emb_dimension
        self._all_item_semantic_embeddings = all_item_semantic_embeddings
        if self._item_semantic_embed:
            self._item_sematic_emb_proj_module: torch.nn.Module = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(
                        in_features=item_semantic_emb_dimension,
                        out_features=dot_product_dimension,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            self._item_semantic_emb_reconstruct_module = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(
                        in_features=dot_product_dimension,
                        out_features=item_semantic_emb_dimension,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            self._reconstruction_loss = torch.nn.MSELoss()
        self._eps: float = eps

    def forward(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (1/B, X, item_embedding_dim,) x float where B is the batch size.
            kwargs: str-keyed tensors. Implementation-specific.

        Returns:
            Tuple of (
                (1/B, X, item_dot_product_groups, dot_product_embedding_dim) x float,
                str-keyed aux_losses,
            ).
        """
        split_item_embeddings = self._item_emb_proj_module(input_embeddings).reshape(
            input_embeddings.size()[:-1]
            + (
                self._item_emb_based_dot_product_groups,
                self._dot_product_dimension,
            )
        )

        aux_losses = {}
        if self._item_semantic_embed:
            item_semantic_embedding = self._item_sematic_emb_proj_module(self._all_item_semantic_embeddings)
            reconstructed_embedding = self._item_semantic_emb_reconstruct_module(item_semantic_embedding)
            if self.training:
                reconstruction_loss = self._reconstruction_loss(
                    reconstructed_embedding, self._all_item_semantic_embeddings
                )
                aux_losses["reconstruction_loss"] = reconstruction_loss
                l2_norm = (item_semantic_embedding * item_semantic_embedding).sum(-1).mean()
                aux_losses["item_semantic_emb_l2_norm"] = l2_norm

            item_semantic_embedding = item_semantic_embedding.unsqueeze(1).unsqueeze(0)

            split_item_embeddings = torch.cat(
                (split_item_embeddings, item_semantic_embedding), dim=2
            )

        if self._dot_product_l2_norm:
            split_item_embeddings = split_item_embeddings / torch.clamp(
                torch.linalg.norm(
                    split_item_embeddings,
                    ord=None,
                    dim=-1,
                    keepdim=True,
                ),
                min=self._eps,
            )
        return split_item_embeddings, aux_losses
