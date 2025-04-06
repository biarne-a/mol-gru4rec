import re
from typing import Dict
import os
import glob

import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from config.config import Config
from train.string_lookup import StringLookup


class MovieDataset(Dataset):
    def __init__(self, feature_description, tfrecord_files, parse_fn, nb_examples):
        self.feature_description = feature_description
        self.tfrecord_files = tfrecord_files
        self.raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
        self.parse_fn = parse_fn
        self.nb_examples = nb_examples

    def __len__(self):
        return self.nb_examples

    def __getitem__(self, idx):
        raw_example = next(iter(self.raw_dataset.skip(idx).take(1)))
        example = tf.io.parse_single_example(raw_example, self.feature_description)
        example = {k: v.numpy() for k, v in example.items()}
        if self.parse_fn:
            example = self.parse_fn(example)
        return example

class Data:
    def __init__(
        self,
        train_ds: DataLoader,
        nb_train: int,
        nb_train_batches: int,
        val_ds: DataLoader,
        nb_val: int,
        nb_val_batches: int,
        test_ds: DataLoader,
        nb_test: int,
        nb_test_batches: int,
        unique_train_movie_id_counts: Dict[str, int],
        movie_id_lookup: StringLookup,
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
        self.unique_train_movie_id_counts = unique_train_movie_id_counts
        self.movie_id_lookup = movie_id_lookup

    @property
    def vocab_size(self):
        return len(self.movie_id_lookup)


def _read_unique_train_movie_id_counts(bucket_dir):
    unique_train_movie_id_counts = {}
    with open(f"{bucket_dir}/vocab/train_movie_counts.txt-00000-of-00001") as f:
        for line in f.readlines():
            match = re.match(r"^\(([0-9]+), ([0-9]+)\)$", line.strip())
            movie_id = match.groups()[0]
            count = int(match.groups()[1])
            unique_train_movie_id_counts[movie_id] = count
    return unique_train_movie_id_counts

def _get_file_list(config: Config, dataset_type: str):
    file_pattern = f"{config.data_dir}/tfrecords/{config.dataset_name}/{dataset_type}/*.gz"
    return glob.glob(file_pattern)

def get_data(config: Config) -> Data:
    unique_train_movie_id_counts = _read_unique_train_movie_id_counts(config.data_dir)

    movie_id_vocab = list(unique_train_movie_id_counts.keys()) + config.model_config.get_special_tokens()
    movie_id_lookup = StringLookup(movie_id_vocab)

    train_features_description = config.model_config.get_train_features_description()
    val_and_test_features_description = config.model_config.get_val_and_test_features_description()

    train_files = _get_file_list(config, "train")
    val_files = _get_file_list(config, "val")
    test_files = _get_file_list(config, "test")

    train_parse_fn = config.model_config.get_parse_sample_fn(movie_id_lookup)
    val_and_test_parse_fn = config.model_config.get_parse_sample_fn(movie_id_lookup)

    nb_train = config.model_config.nb_train
    nb_val = config.model_config.nb_val
    nb_test = config.model_config.nb_test

    nb_train_batches = nb_train // config.batch_size
    nb_val_batches = nb_val // config.batch_size
    nb_test_batches = nb_test // config.batch_size

    train_ds = DataLoader(MovieDataset(train_features_description, train_files, train_parse_fn, nb_train), batch_size=config.batch_size)
    val_ds = DataLoader(MovieDataset(val_and_test_features_description, val_files, val_and_test_parse_fn, nb_val), batch_size=config.batch_size)
    test_ds = DataLoader(MovieDataset(val_and_test_features_description, test_files, val_and_test_parse_fn, nb_test), batch_size=config.batch_size)

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
    )
