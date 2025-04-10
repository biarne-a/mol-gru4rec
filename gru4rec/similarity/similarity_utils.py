# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# forked from facebookresearch/generative-recommenders @ 6c61e25 and updated
# to match MoL implementations on public datasets.

# Defines utility functions used to create Mixture-of-Logits learned similarity functions. Used by
# - Revisiting Neural Retrieval on Accelerators (KDD'23)
# - Retrieval with Learned Similarities (RAILS).

from typing import Tuple, Optional, List

import torch

from config import Config
from config.similarity_config import SimilarityConfig
from gru4rec.similarity.cosine_similarity import CosineSimilarity
from gru4rec.similarity.dot_product_similarity import DotProductSimilarity
from gru4rec.similarity.mol.layers import GeGLU, SwiGLU
from gru4rec.similarity.mol.mol_similarity import MoLSimilarity, SoftmaxDropoutCombiner
from gru4rec.similarity.mol.query_embeddings_fns import RecoMoLQueryEmbeddingsFn
from gru4rec.similarity.mol.item_embeddings_fns import RecoMoLItemEmbeddingsFn


def init_mlp_xavier_weights_zero_bias(m) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0.0)


def create_mol_interaction_module(config: Config) -> Tuple[MoLSimilarity, str]:
    """
    Gin wrapper for creating MoL learned similarity.
    """
    similarity_config = config.model_config.similarity_config
    mol_module = MoLSimilarity(
        query_embedding_dim=similarity_config.query_embedding_dim,
        item_embedding_dim=similarity_config.item_embedding_dim,
        dot_product_dimension=similarity_config.dot_product_dimension,
        query_dot_product_groups=similarity_config.query_dot_product_groups,
        item_dot_product_groups=similarity_config.item_dot_product_groups,
        temperature=0.05,
        dot_product_l2_norm=None,
        query_embeddings_fn=RecoMoLQueryEmbeddingsFn(
            query_embedding_dim=similarity_config.query_embedding_dim,
            query_dot_product_groups=similarity_config.query_dot_product_groups,
            dot_product_dimension=similarity_config.dot_product_dimension,
            dot_product_l2_norm=None,
            proj_fn=lambda input_dim, output_dim: (
                torch.nn.Sequential(
                    torch.nn.Dropout(p=similarity_config.query_dropout_rate),
                    (
                        GeGLU(
                            in_features=input_dim,
                            out_features=similarity_config.query_hidden_dim,
                        )
                        if similarity_config.query_nonlinearity == "geglu"
                        else SwiGLU(
                            in_features=input_dim,
                            out_features=similarity_config.query_hidden_dim,
                        )
                    ),
                    torch.nn.Linear(
                        in_features=similarity_config.query_hidden_dim,
                        out_features=output_dim,
                    ),
                )
                if similarity_config.query_hidden_dim > 0
                else torch.nn.Sequential(
                    torch.nn.Dropout(p=similarity_config.query_dropout_rate),
                    torch.nn.Linear(
                        in_features=input_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            ),
            uid_embed=similarity_config.uid_embed,
            uid_dropout_rate=similarity_config.uid_dropout_rate,
            uid_embedding_level_dropout=similarity_config.uid_embedding_level_dropout,
            eps=1e-6,
        ),
        item_embeddings_fn=RecoMoLItemEmbeddingsFn(
            item_embedding_dim=similarity_config.item_embedding_dim,
            item_dot_product_groups=similarity_config.item_dot_product_groups,
            dot_product_dimension=similarity_config.dot_product_dimension,
            dot_product_l2_norm=None,
            proj_fn=lambda input_dim, output_dim: (
                torch.nn.Sequential(
                    torch.nn.Dropout(p=similarity_config.item_dropout_rate),
                    (
                        GeGLU(
                            in_features=input_dim,
                            out_features=similarity_config.item_hidden_dim,
                        )
                        if similarity_config.item_nonlinearity == "geglu"
                        else SwiGLU(in_features=input_dim, out_features=similarity_config.item_hidden_dim)
                    ),
                    torch.nn.Linear(
                        in_features=similarity_config.item_hidden_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
                if similarity_config.item_hidden_dim > 0
                else torch.nn.Sequential(
                    torch.nn.Dropout(p=similarity_config.item_dropout_rate),
                    torch.nn.Linear(
                        in_features=input_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            ),
            eps=1e-6,
        ),
        gating_query_only_partial_fn=lambda input_dim, output_dim: (
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=similarity_config.gating_query_hidden_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=similarity_config.gating_query_hidden_dim,
                    out_features=output_dim,
                    bias=False,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
            if similarity_config.gating_query_fn
            else None
        ),
        gating_item_only_partial_fn=lambda input_dim, output_dim: (
            torch.nn.Sequential(
                torch.nn.Dropout(p=similarity_config.gating_item_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=similarity_config.gating_item_hidden_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=similarity_config.gating_item_hidden_dim,
                    out_features=output_dim,
                    bias=False,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
            if similarity_config.gating_item_fn
            else None
        ),
        gating_qi_partial_fn=lambda input_dim, output_dim: (
            torch.nn.Sequential(
                torch.nn.Dropout(p=similarity_config.gating_qi_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=similarity_config.gating_qi_hidden_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=similarity_config.gating_qi_hidden_dim,
                    out_features=output_dim,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
            if similarity_config.gating_qi_hidden_dim > 0
            else torch.nn.Sequential(
                torch.nn.Dropout(p=similarity_config.gating_qi_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=output_dim,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
        ),
        gating_combination_type=similarity_config.gating_combination_type,
        gating_normalization_fn=lambda _: SoftmaxDropoutCombiner(
            dropout_rate=0.1, eps=1e-6
        ),
        eps=1e-6,
        autocast_bf16=False,
    )
    # interaction_module_debug_str = (
    #     f"MoL-{query_dot_product_groups}x{item_dot_product_groups}x{dot_product_dimension}"
    #     + f"-t{temperature}-d{softmax_dropout_rate}"
    #     + f"{'-l2' if dot_product_l2_norm else ''}"
    #     + (
    #         f"-q{query_hidden_dim}d{query_dropout_rate}{query_nonlinearity}"
    #         if query_hidden_dim > 0
    #         else f"-cd{query_dropout_rate}"
    #     )
    #     + (
    #         f"-{item_hidden_dim}d{item_dropout_rate}{item_nonlinearity}"
    #         if item_hidden_dim > 0
    #         else f"-id{item_dropout_rate}"
    #     )
    #     + (f"-gq{gating_query_hidden_dim}" if gating_query_fn else "")
    #     + (
    #         f"-gi{gating_item_hidden_dim}d{gating_item_dropout_rate}"
    #         if gating_item_fn
    #         else ""
    #     )
    #     + f"-gqi{gating_qi_hidden_dim}d{gating_qi_dropout_rate}-x-{gating_combination_type}"
    # )
    # if uid_embedding_hash_sizes is not None:
    #     interaction_module_debug_str += (
    #         f"-uids{'-'.join([str(x) for x in uid_embedding_hash_sizes])}"
    #     )
    #     if uid_dropout_rate > 0.0:
    #         interaction_module_debug_str += f"d{uid_dropout_rate}"
    #     if uid_embedding_level_dropout:
    #         interaction_module_debug_str += "-el"
    interaction_module_debug_str = ""
    return mol_module, interaction_module_debug_str


def get_similarity_module(config: Config) -> Tuple[torch.nn.Module, str]:
    similarity_config = config.model_config.similarity_config
    if similarity_config.type == "dot_product":
        return DotProductSimilarity()
    if similarity_config.type == "cosine":
        return CosineSimilarity()
    if similarity_config.type == "mol":
        interaction_module, interaction_module_debug_str = create_mol_interaction_module(config)
        print(f"Interaction module: {interaction_module_debug_str}")
        return interaction_module
    raise ValueError(f"Unknown similarity module {similarity_config.type}")
