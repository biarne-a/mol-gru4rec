import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config.config import Config
from train.datasets import Data


def save_history(history: tf.keras.callbacks.History, config: Config):
    output_file = f"{config.results_dir}/history_training.p"
    pickle.dump(history.history, tf.io.gfile.GFile(output_file, "wb"))


def save_predictions(config: Config, data: Data, model: tf.keras.models.Model, k: int = 10):
    nb_test_batches = data.nb_test_batches
    label_column = config.model_config.label_column
    local_filename = f"{config.results_dir}/predictions.csv"
    with tf.io.gfile.GFile(local_filename, "w") as fileh:
        columns = ["label"] + [f"output_{i}" for i in range(k)]
        header = ",".join(columns)
        fileh.write(f"{header}\n")
        i_batch = 0
        for batch in tqdm(data.test_ds, total=nb_test_batches):
            logits = model.predict_on_batch(batch)
            top_indices = tf.math.top_k(logits, k=k).indices
            top_predictions = data.reverse_movie_id_lookup(top_indices).numpy().reshape((-1, k))
            y_true = batch[label_column]
            y_true = data.reverse_movie_id_lookup(y_true)
            y_true = np.reshape(y_true.numpy(), (config.batch_size, -1))
            predictions_numpy = np.concatenate((y_true, top_predictions), axis=1)
            np.savetxt(fileh, predictions_numpy.astype(str), fmt="%s", delimiter=",")
            i_batch += 1
            if i_batch == nb_test_batches:
                break
            fileh.flush()
    return local_filename
