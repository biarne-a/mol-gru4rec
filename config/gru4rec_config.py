import copy
from typing import Any

from config.similarity_config import SimilarityConfig


class Gru4RecConfig:
    def __init__(
        self,
        embedding_dim=64,
        dropout_p_embed: float = 0.0,
        dropout_p_gru: float = 0.0,
        aux_loss_weights: dict[str, float] | None = None,
        similarity_config: dict[str, Any] | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_gru = dropout_p_gru
        self.aux_loss_weights = aux_loss_weights or {}
        self.similarity_config = SimilarityConfig.from_dict(similarity_config or {})

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        return Gru4RecConfig(**json_object)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
