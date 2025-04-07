import re
from typing import Dict
import glob

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from config.config import Config
from train.string_lookup import StringLookup


class Data:
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        unique_train_movie_id_counts: Dict[str, int],
        movie_id_lookup: StringLookup,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.unique_train_movie_id_counts = unique_train_movie_id_counts
        self.movie_id_lookup = movie_id_lookup


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
    file_pattern = f"{config.data_dir}/parquets/{config.dataset_name}/{dataset_type}/*.parquet"
    return glob.glob(file_pattern)

def get_data(config: Config) -> Data:
    unique_train_movie_id_counts = _read_unique_train_movie_id_counts(config.data_dir)

    movie_id_vocab = list(unique_train_movie_id_counts.keys())
    movie_id_lookup = StringLookup(movie_id_vocab)

    train_files = _get_file_list(config, "train")
    val_files = _get_file_list(config, "val")
    test_files = _get_file_list(config, "test")

    train_dataloader = _build_dataloader(config, movie_id_lookup, train_files)
    val_dataloader = _build_dataloader(config, movie_id_lookup, val_files)
    test_dataloader = _build_dataloader(config, movie_id_lookup, test_files)

    return Data(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        unique_train_movie_id_counts,
        movie_id_lookup,
    )


def _build_dataloader(config, movie_id_lookup, parquet_files):
    def _parse_sample(x):
        return {
            "label": torch.tensor(movie_id_lookup.lookup(x["label"]), dtype=torch.long),
            "input_ids": torch.tensor(movie_id_lookup.lookup(x["input_ids"]), dtype=torch.long),
            "input_mask": torch.tensor(x["input_mask"], dtype=torch.long),
        }

    dataset = next(
        iter(load_dataset("parquet", data_files=parquet_files, streaming=True).values())
    )
    dataset = dataset.map(_parse_sample, batched=False)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    return dataloader
