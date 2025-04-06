from typing import Dict

import tensorflow as tf

from train.string_lookup import StringLookup


def get_features_description() -> Dict[str, tf.io.FixedLenFeature]:
    return {
        "input_ids": tf.io.FixedLenFeature([200], tf.int64),
        "input_mask": tf.io.FixedLenFeature([200], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }


def get_parse_sample_fn(movie_id_lookup: StringLookup):
    def _parse_sample(x):
        return {
            "label": movie_id_lookup.lookup(x["label"]),
            "input_ids": movie_id_lookup.lookup(x["input_ids"]),
            "input_mask": x["input_mask"]
        }
    return _parse_sample
