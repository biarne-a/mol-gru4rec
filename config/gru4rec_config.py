import copy


class Gru4RecConfig:
    def __init__(
        self,
        embedding_dim=64,
        dropout_p_embed: float = 0.0,
        dropout_p_gru: float = 0.0,
        **kwargs
    ):
        """
        Builds a BertConfig.

        :param hidden_size: Size of the encoder layers and the pooler layer.
        :param inner_dim: The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        """
        self.embedding_dim = embedding_dim
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_gru = dropout_p_gru

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        return Gru4RecConfig(**json_object)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    @property
    def label_column(self):
        return "label"
