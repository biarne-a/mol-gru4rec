import copy

from config.config import ModelConfig


class Gru4RecConfig(ModelConfig):
    def __init__(
        self,
        hidden_size=768,
        nb_train=988129,
        nb_val=6040,
        nb_test=6040,
        **kwargs
    ):
        """
        Builds a BertConfig.

        :param hidden_size: Size of the encoder layers and the pooler layer.
        :param inner_dim: The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        """
        self.hidden_size = hidden_size
        self.nb_train = nb_train
        self.nb_val = nb_val
        self.nb_test = nb_test

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        return Gru4RecConfig(**json_object)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def get_train_features_description(self):
        from gru4rec.dataset_helpers import get_features_description
        return get_features_description()

    def get_val_and_test_features_description(self):
        from gru4rec.dataset_helpers import get_features_description
        return get_features_description()

    def get_parse_sample_fn(
      self, movie_id_lookup
    ):
        from gru4rec.dataset_helpers import get_parse_sample_fn
        return get_parse_sample_fn(movie_id_lookup)

    def build_model(self, data):
        from gru4rec.gru4rec_torch_model import Gru4RecModel
        gru4rec_config = self.to_dict()
        gru4rec_config.pop("nb_train")
        gru4rec_config.pop("nb_val")
        gru4rec_config.pop("nb_test")
        return Gru4RecModel(data.vocab_size, **gru4rec_config)

    def get_special_tokens(self):
      return []

    @property
    def label_column(self):
        return "label"
