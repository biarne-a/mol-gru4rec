import sys
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Recall

from config.config import Config
from train.datasets import get_data, Data
from train.save_results import save_history, save_predictions


def _debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def set_seed():
    # for tf.random
    tf.random.set_seed(Config.SEED)
    # for numpy.random
    np.random.seed(Config.SEED)
    # for built-in random
    random.seed(Config.SEED)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(Config.SEED)


def _get_model_local_save_filepath(config: Config) -> str:
    local_savedir = f"data/results/{config.dataset_name}"
    os.makedirs(local_savedir, exist_ok=True)
    return f"{local_savedir}/model.weights.h5"


def _build_optimizer(config, data):
    leaarning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=data.nb_train_batches * 40,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False,
    )
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=leaarning_rate_schedule,
                                          epsilon=1e-6,
                                          global_clipnorm=5.0)
    return opt


def _compile_model(model, config, data):
  optimizer = _build_optimizer(config, data)
  model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[
          Recall(top_k=10),
      ],
      run_eagerly=_debugger_is_active(),
  )


def run_training(config: Config):
    os.makedirs(config.results_dir, exist_ok=True)

    data = get_data(config)
    model = config.model_config.build_model(data)
    _compile_model(model, config, data)
    local_save_filepath = _get_model_local_save_filepath(config)
    history = model.fit(
        x=data.train_ds,
        epochs=2,#1_000,
        steps_per_epoch=20,#data.nb_train_batches,
        validation_data=data.val_ds,
        validation_steps=300,#data.nb_val_batches,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir="logs", update_freq=100),
            tf.keras.callbacks.EarlyStopping(monitor="val_recall_at_10", mode="max", patience=2),
            tf.keras.callbacks.ModelCheckpoint(
              local_save_filepath, monitor="val_recall_at_10", mode="max", 
              save_best_only=True, save_weights_only=True, verbose=2
            ),
        ],
        verbose=1,
    )
    distant_save_filepath = f"{config.results_dir}/model.weights.h5"
    if local_save_filepath != distant_save_filepath:
      tf.io.gfile.copy(
        local_save_filepath, distant_save_filepath, overwrite=True
    )
    save_history(history, config)
    run_evaluation(config, data)


def run_evaluation(config: Config, data: Data):
    data = get_data(config)
    local_save_filepath = _get_model_local_save_filepath(config)
    model = config.model_config.build_model(data)
    _compile_model(model, config, data)
    model.load_weights(local_save_filepath)
    model.evaluate(data.test_ds, steps=data.nb_test_batches)
    save_predictions(config, data, model)
