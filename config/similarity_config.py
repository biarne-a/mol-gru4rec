import copy


class SimilarityConfig:
    def __init__(
        self,
        type: str,

        query_embedding_dim: int,
        item_embedding_dim: int,
        dot_product_dimension: int,
        query_dot_product_groups: int,
        query_dropout_rate: float,
        query_hidden_dim: int,

        # item_dot_product_groups: int,
        # item_dropout_rate: float,
        # item_hidden_dim: int,
        # temperature: float,

        gating_query_fn: bool,
        gating_item_fn: bool,
        gating_query_hidden_dim: int,
        gating_qi_hidden_dim: int,
        gating_item_hidden_dim: int,
        gating_item_dropout_rate: float,
        gating_qi_dropout_rate: float,
        gating_combination_type: str,

        uid_embed: bool = False,
        uid_dropout_rate: float = 0.5,
        uid_embedding_level_dropout: bool = False,
        # softmax_dropout_rate: float,
        # bf16_training: bool,
        # dot_product_l2_norm: bool = True,
        query_nonlinearity: str = "geglu",
        # item_nonlinearity: str = "geglu",
        # eps: float = 1e-6,
    ):
        self.type = type
        self.query_embedding_dim = query_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.dot_product_dimension = dot_product_dimension
        self.query_dot_product_groups = query_dot_product_groups
        self.query_dropout_rate = query_dropout_rate
        self.query_hidden_dim = query_hidden_dim

        self.gating_query_fn = gating_query_fn
        self.gating_item_fn = gating_item_fn
        self.gating_query_hidden_dim = gating_query_hidden_dim
        self.gating_qi_hidden_dim = gating_qi_hidden_dim
        self.gating_item_hidden_dim = gating_item_hidden_dim
        self.gating_item_dropout_rate = gating_item_dropout_rate
        self.gating_qi_dropout_rate = gating_qi_dropout_rate
        self.gating_combination_type = gating_combination_type

        # self.query_embedding_dim = query_embedding_dim
        # self.item_embedding_dim = item_embedding_dim
        # self.dot_product_dimension = dot_product_dimension
        # self.query_dot_product_groups = query_dot_product_groups
        # self.item_dot_product_groups = item_dot_product_groups
        # self.temperature = temperature
        # self.query_dropout_rate = query_dropout_rate
        # self.query_hidden_dim = query_hidden_dim
        # self.item_dropout_rate = item_dropout_rate
        # self.item_hidden_dim = item_hidden_dim
        # self.softmax_dropout_rate = softmax_dropout_rate
        # self.bf16_training = bf16_training
        # self.dot_product_l2_norm = dot_product_l2_norm
        self.query_nonlinearity = query_nonlinearity
        # self.item_nonlinearity = item_nonlinearity
        self.uid_embed = uid_embed
        self.uid_dropout_rate = uid_dropout_rate
        self.uid_embedding_level_dropout = uid_embedding_level_dropout
        # self.eps = eps

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        return SimilarityConfig(**json_object)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
