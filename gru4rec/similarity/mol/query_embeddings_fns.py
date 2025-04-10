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

"""
Defines functions to generate query-side embeddings for MoL.
"""

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


class RecoMoLQueryEmbeddingsFn(MoLEmbeddingsFn):
    """
    Generates P_Q query-side embeddings for MoL based on input embeddings and other
    optional tensors for recommendation models. Tested for sequential retrieval
    scenarios.

    The current implementation accesses user_ids associated with the query from
    `user_ids' in kwargs.
    """

    def __init__(
        self,
        query_embedding_dim: int,
        query_dot_product_groups: int,
        dot_product_dimension: int,
        dot_product_l2_norm: bool,
        proj_fn: Callable[[int, int], torch.nn.Module],
        eps: float,
        uid_embed: bool,
        uid_dropout_rate: float,
        uid_embedding_level_dropout: bool = False,
    ) -> None:
        super().__init__()
        self._uid_embed: bool = uid_embed
        # The uid embedding should be included in the dot product groups.
        self._query_emb_based_dot_product_groups: int = query_dot_product_groups - int(uid_embed)
        self._query_emb_proj_module: torch.nn.Module = proj_fn(
            query_embedding_dim,
            dot_product_dimension * self._query_emb_based_dot_product_groups,
        )
        self._dot_product_dimension: int = dot_product_dimension
        self._dot_product_l2_norm: bool = dot_product_l2_norm
        if self._uid_embed:
            # TODO Complete that
            user_id_vocab_len = 100_000
            self._uid_embedding = torch.nn.Embedding(
                user_id_vocab_len, dot_product_dimension, padding_idx=0
            )
        self._uid_dropout_rate: float = uid_dropout_rate
        self._uid_embedding_level_dropout: bool = uid_embedding_level_dropout
        self._eps: float = eps

    def forward(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, query_embedding_dim,) x float where B is the batch size.
            kwargs: str-keyed tensors. Implementation-specific.

        Returns:
            Tuple of (
                (B, query_dot_product_groups, dot_product_embedding_dim) x float,
                str-keyed aux_losses,
            ).
        """
        split_query_embeddings = self._query_emb_proj_module(input_embeddings).reshape(
            (
                input_embeddings.size(0),
                self._query_emb_based_dot_product_groups,
                self._dot_product_dimension,
            )
        )

        aux_losses: Dict[str, torch.Tensor] = {}

        if self._uid_embed:
            uid_embedding = self._uid_embedding(kwargs["user_id"])

            if self.training:
                l2_norm = (uid_embedding * uid_embedding).sum(-1).mean()
                aux_losses["uid_embedding_l2_norm"] = l2_norm

            if self._uid_dropout_rate > 0.0:
                if self._uid_embedding_level_dropout:
                    # conditionally dropout the entire embedding.
                    if self.training:
                        uid_dropout_mask = (
                            torch.rand(
                                uid_embedding.size()[:-1],
                                device=uid_embedding.device,
                            )
                            > self._uid_dropout_rate
                        )
                        uid_embedding = (
                            uid_embedding
                            * uid_dropout_mask.unsqueeze(-1)
                            / (1.0 - self._uid_dropout_rate)
                        )
                else:
                    uid_embedding = F.dropout(
                        uid_embedding,
                        p=self._uid_dropout_rate,
                        training=self.training,
                    )
            uid_embedding = uid_embedding.unsqueeze(1)
            split_query_embeddings = torch.cat(
                [split_query_embeddings] + uid_embedding, dim=1
            )

        if self._dot_product_l2_norm:
            split_query_embeddings = split_query_embeddings / torch.clamp(
                torch.linalg.norm(
                    split_query_embeddings,
                    ord=None,
                    dim=-1,
                    keepdim=True,
                ),
                min=self._eps,
            )
        return split_query_embeddings, aux_losses
