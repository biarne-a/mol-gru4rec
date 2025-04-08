import json


class Config:
    SEED = 42

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        config_name: str,
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.config_name = config_name
        config_filename = f"config/{config_name}.json"
        json_model_config = json.load(open(config_filename, "r"))
        self.batch_size = json_model_config.pop("batch_size")
        self.epochs = json_model_config.pop("epochs")
        self.learning_rate = json_model_config.pop("learning_rate")
        self.val_every_n_steps = json_model_config.pop("val_every_n_steps")
        self.early_stopping_patience = json_model_config.pop("early_stopping_patience")
        self.early_stopping_metric = json_model_config.pop("early_stopping_metric")
        model_config_cls = self._fetch_class(json_model_config.pop("model_config_name"))
        self.model_config = model_config_cls.from_dict(json_model_config)

    def _fetch_class(self, class_name):
        try:
            from config.gru4rec_config import Gru4RecConfig

            return locals()[class_name]
        except KeyError:
            raise ValueError(f"Class '{class_name}' not found")

    @property
    def results_dir(self):
        return f"{self.data_dir}/results/{self.config_name}"
