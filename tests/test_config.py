import unittest
import os
from config.config import Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a sample JSON file for testing
        self.test_json_filename = "config/gru4rec_config_ml-1m_64_test.json"
        self.test_json_content = {
            "model_config_name": "Gru4RecConfig",
            "batch_size": 128,
            "epochs": 10,
            "val_every_n_steps": 100,
            "early_stopping_patience": 2,
            "early_stopping_metric": "val_loss",
            "learning_rate": 0.001,
            "model_config": {
                "embedding_dim": 64,
                "dropout_p_embed": 0.1,
                "dropout_p_gru": 0.2,
                "similarity_config": {
                    "type": "dot_product"
                }
            }
        }
        os.makedirs("config", exist_ok=True)
        with open(self.test_json_filename, "w") as f:
            import json
            json.dump(self.test_json_content, f)

    def tearDown(self):
        # Clean up the test JSON file
        if os.path.exists(self.test_json_filename):
            os.remove(self.test_json_filename)

    def test_config_loading(self):
        # Load the config
        config = Config(data_dir="data", dataset_name="ml-1m", config_name="gru4rec_config_ml-1m_64_test")

        # Check top-level attributes
        self.assertEqual(config.batch_size, self.test_json_content["batch_size"])
        self.assertEqual(config.epochs, self.test_json_content["epochs"])
        self.assertEqual(config.learning_rate, self.test_json_content["learning_rate"])
        self.assertEqual(config.val_every_n_steps, self.test_json_content["val_every_n_steps"])
        self.assertEqual(config.early_stopping_patience, self.test_json_content["early_stopping_patience"])
        self.assertEqual(config.early_stopping_metric, self.test_json_content["early_stopping_metric"])

        # Check model_config attributes
        model_config = config.model_config
        self.assertEqual(model_config.embedding_dim, self.test_json_content["model_config"]["embedding_dim"])
        self.assertEqual(model_config.dropout_p_embed, self.test_json_content["model_config"]["dropout_p_embed"])
        self.assertEqual(model_config.dropout_p_gru, self.test_json_content["model_config"]["dropout_p_gru"])

        # Check similarity_config attributes
        similarity_config = model_config.similarity_config
        self.assertEqual(similarity_config.type, self.test_json_content["model_config"]["similarity_config"]["type"])


if __name__ == "__main__":
    unittest.main()
