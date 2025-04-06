import re
from typing import Dict

import tensorflow as tf

from config.config import Config


class Data:
    def __init__(
        self,
        train_ds: tf.data.Dataset,
        nb_train: int,
        nb_train_batches: int,
        val_ds: tf.data.Dataset,
        nb_val: int,
        nb_val_batches: int,
        test_ds: tf.data.Dataset,
        nb_test: int,
        nb_test_batches: int,
        movie_id_counts: Dict[str, int],
        movie_id_lookup: tf.keras.layers.StringLookup,
        reverse_movie_id_lookup: tf.keras.layers.StringLookup,
    ):
        self.train_ds = train_ds
        self.nb_train = nb_train
        self.nb_train_batches = nb_train_batches
        self.val_ds = val_ds
        self.nb_val = nb_val
        self.nb_val_batches = nb_val_batches
        self.test_ds = test_ds
        self.nb_test = nb_test
        self.nb_test_batches = nb_test_batches
        self.movie_id_counts = movie_id_counts
        self.movie_id_lookup = movie_id_lookup
        self.reverse_movie_id_lookup = reverse_movie_id_lookup

    @property
    def vocab_size(self):
        return self.movie_id_lookup.vocabulary_size()


def _read_unique_train_movie_id_counts(bucket_dir):
    with tf.io.gfile.GFile(f"{bucket_dir}/vocab/train_movie_counts.txt-00000-of-00001") as f:
        unique_train_movie_id_counts = {}
        for line in f.readlines():
            match = re.match("^\(([0-9]+), ([0-9]+)\)$", line.strip())  # noqa: W605
            movie_id = match.groups()[0]
            count = int(match.groups()[1])
            unique_train_movie_id_counts[movie_id] = count
    return unique_train_movie_id_counts


def _get_dataset_from_files(config: Config, dataset_type: str):
    filenames = f"{config.data_dir}/tfrecords/{config.dataset_name}/{dataset_type}/*.gz"
    dataset = tf.data.Dataset.list_files(filenames, seed=Config.SEED)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=8,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    return dataset


def get_data(config: Config) -> Data:
    unique_train_movie_id_counts = _read_unique_train_movie_id_counts(config.data_dir)

    train_ds = _get_dataset_from_files(config, "train")
    val_ds = _get_dataset_from_files(config, "val")
    test_ds = _get_dataset_from_files(config, "test")

    movie_id_vocab = (
      list(unique_train_movie_id_counts.keys()) + 
      config.model_config.get_special_tokens()
    )
    movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab)
    reverse_movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab, invert=True)

    train_features_description = config.model_config.get_train_features_description()
    val_and_test_features_description = config.model_config.get_val_and_test_features_description()

    train_parse_fn = config.model_config.get_parse_sample_fn(
      train_features_description, movie_id_lookup, training=True
    )
    val_and_test_parse_fn = config.model_config.get_parse_sample_fn(
      val_and_test_features_description, movie_id_lookup, training=False
    )

    nb_train = config.model_config.nb_train
    nb_val = config.model_config.nb_val
    nb_test = config.model_config.nb_test

    nb_train_batches = nb_train // config.batch_size
    nb_val_batches = nb_val // config.batch_size
    nb_test_batches = nb_test // config.batch_size

    train_ds = (
      train_ds.map(train_parse_fn)
              .batch(config.batch_size)
              .take(nb_train_batches)
              .repeat()
    )
    val_ds = (
        val_ds.map(val_and_test_parse_fn)
              .batch(config.batch_size)
              .take(nb_val_batches)
              .repeat()
    )
    test_ds = (
        test_ds.map(val_and_test_parse_fn)
               .batch(config.batch_size)
               .take(nb_test_batches)
               .repeat()
    )

    return Data(
        train_ds,
        nb_train,
        nb_train_batches,
        val_ds,
        nb_val,
        nb_val_batches,
        test_ds,
        nb_test,
        nb_test_batches,
        unique_train_movie_id_counts,
        movie_id_lookup,
        reverse_movie_id_lookup
    )
