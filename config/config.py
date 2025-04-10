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
        json_config = json.load(open(config_filename, "r"))

        from config import Gru4RecConfig

        self.model_config = Gru4RecConfig.from_dict(json_config.pop("model_config"))
        self.batch_size = json_config.pop("batch_size")
        self.epochs = json_config.pop("epochs")
        self.learning_rate = json_config.pop("learning_rate")
        self.val_every_n_steps = json_config.pop("val_every_n_steps")
        self.early_stopping_patience = json_config.pop("early_stopping_patience")
        self.early_stopping_metric = json_config.pop("early_stopping_metric")

    @property
    def results_dir(self):
        return f"{self.data_dir}/results/{self.config_name}"
