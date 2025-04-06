import random
# from enum import Enum
from typing import Any, Dict, List, Tuple, Union, Optional

import tensorflow as tf
import apache_beam as beam
from apache_beam.pvalue import PCollection
import pyarrow as pa


class SampleType: #(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


def _sort_views_by_timestamp(group) -> List[int]:
    views = group[1]
    views.sort(key=lambda x: x[-1])
    return [v[0] for v in views]


def _generate_examples_from_complete_sequences(
    complete_sequence: List[int],
    max_context_len: int,
    proportion_sliding_window: Optional[float] = None,
    sliding_window_step_size_override: Optional[int] = None,
):
    """
    Generate user sequences from a single complete user sequence using a sliding window.

    :param complete_sequence: The complete sequence to generate examples from (A list of ids).
    :param max_context_len: The maximum length of the context.
    :param proportion_sliding_window: The proportion of the complete user sequence that will be used as the size of
    the step in the sliding window used to create the training sequences.
    :param sliding_window_step_size_override: Ability to override the step size proportion with an explicit step size
    number.

    :return: examples: Generated examples from this single timeline.
    """
    def _get_sliding_window_step_size(
        max_context_len: int,
        proportion_sliding_window: float = None,
        sliding_window_step_size_override: int = None,
    ):
        if sliding_window_step_size_override:
            return sliding_window_step_size_override
        if proportion_sliding_window:
            return int(proportion_sliding_window * max_context_len)
        return max_context_len

    sliding_window_step_size = _get_sliding_window_step_size(
        max_context_len, proportion_sliding_window, sliding_window_step_size_override
    )

    def _get_new_example(complete_sequence, start_idx, end_idx, sample_type):
        input_ids = complete_sequence[start_idx:end_idx]
        return {
            "input_ids": input_ids,
            "sample_type": sample_type,
        }

    examples = []
    complete_sequence_len = len(complete_sequence)

    # The last 2 tokens of each sequence is for validation and testing
    # Add test sequence
    end_idx_test = complete_sequence_len
    start_idx_test = max(0, end_idx_test - max_context_len)
    example = _get_new_example(complete_sequence, start_idx_test, end_idx_test, sample_type=2)
    examples.append(example)

    # Add validation sequence
    end_idx_validation = complete_sequence_len - 1
    start_idx_validation = max(0, end_idx_validation - max_context_len)
    example = _get_new_example(complete_sequence, start_idx_validation, end_idx_validation, sample_type=1)
    examples.append(example)

    # Add train sequences
    end_indexes = list(range(end_idx_validation - 1, 0, -sliding_window_step_size))
    start_indexes = [max(0, idx - max_context_len) for idx in end_indexes]
    for start_idx, end_idx in zip(start_indexes, end_indexes):
        example = _get_new_example(complete_sequence, start_idx, end_idx, sample_type=0)
        examples.append(example)

    return examples


def _prepare_bert4rec_train_samples(
    sample: Dict[str, Any],
    max_context_len: int,
    duplication_factor: int,
    nb_max_masked_ids_per_seq: int,
    mask_ratio: float,
) -> List[Dict[str, Any]]:
    """
    If user complete sequence is shorter than max_context_len, sequence will be padded with 0s.

    """
    import random

    nb_filled_input_ids = len(sample["input_ids"])
    # Pad sequence with 0s.
    input_ids = list(sample["input_ids"])
    input_mask = [1] * nb_filled_input_ids
    while len(input_ids) < max_context_len:
        input_ids.append(0)
        input_mask.append(0)

    all_augmented_samples = []
    for _ in range(duplication_factor):
        nb_ids_to_mask = min(nb_max_masked_ids_per_seq, max(1, int(nb_filled_input_ids * mask_ratio)))

        masked_lm_ids = []
        masked_lm_positions = []
        masked_lm_weights = []

        # Shuffle the positions
        shuffled_id_positions = list(range(nb_filled_input_ids))
        random.shuffle(shuffled_id_positions)

        # And take the required number of masked ids
        for idx in range(nb_ids_to_mask):
            masked_position = shuffled_id_positions[idx]
            masked_lm_id = input_ids[masked_position]
            masked_lm_positions.append(masked_position)
            masked_lm_ids.append(masked_lm_id)
            masked_lm_weights.append(1.0)

        # Pad the masks to obtain a complete sequence up to the maximum allowed
        while len(masked_lm_positions) < nb_max_masked_ids_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        augmented_sample = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_weights": masked_lm_weights,
        }
        all_augmented_samples.append(augmented_sample)

    # Add another augmented sample for last position masking
    masked_lm_position = nb_filled_input_ids - 1
    masked_lm_positions = [masked_lm_position]
    masked_lm_ids = [input_ids[masked_lm_position]]
    masked_lm_weights = [1.0]

    # Pad the masks to obtain a complete sequence up to the maximum allowed
    while len(masked_lm_positions) < nb_max_masked_ids_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

    last_position_masked_sample = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_ids": masked_lm_ids,
        "masked_lm_weights": masked_lm_weights
    }

    # Mask last position to match the val and test sets settings
    all_augmented_samples.append(last_position_masked_sample)
    return all_augmented_samples


def _prepare_bert4rec_val_test_sample(sample: Dict[str, Any], max_context_len: int, **kwargs) -> List[Dict[str, Any]]:
    input_ids = list(sample["input_ids"])
    nb_filled_input_ids = len(input_ids)
    input_mask = [1] * nb_filled_input_ids
    # Pad sequence with 0s.
    while len(input_ids) < max_context_len:
        input_ids.append(0)
        input_mask.append(0)

    masked_lm_position = nb_filled_input_ids - 1
    return [{
        "input_ids": input_ids,
        "input_mask": input_mask,
        "masked_lm_positions":  [masked_lm_position],
        "masked_lm_ids": [input_ids[masked_lm_position]],
        "masked_lm_weights": [1.0]
    }]


def _prepare_gru4rec_samples(sample: Dict[str, Any], max_context_len: int, **kwargs) -> List[Dict[str, Any]]:
    input_ids = sample["input_ids"][:-1]
    label = sample["input_ids"][-1]
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_context_len:
        input_ids.append(0)
        input_mask.append(0)

    return [{
        "input_ids": input_ids,
        "input_mask": input_mask,
        "label": [label],
    }]


def _count_movies_in_ratings(train_samples: PCollection):
    return (
            train_samples
            | "Flatten train samples for count" >> beam.FlatMap(lambda x: x["input_ids"])
            | "Set Movie Id Key" >> beam.Map(lambda x: (x, 1))
            | "Count By Movie Id" >> beam.combiners.Count.PerKey()
            | "Remove 0 counts" >> beam.Filter(lambda x: x[0] != 0)
    )


def _serialize_bert4rec_tfrecords(x):
    import tensorflow as tf

    feature = {
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=x["input_ids"])),
        "input_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=x["input_mask"])),
        "masked_lm_positions": tf.train.Feature(int64_list=tf.train.Int64List(value=x["masked_lm_positions"])),
        "masked_lm_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=x["masked_lm_ids"])),
        "masked_lm_weights": tf.train.Feature(float_list=tf.train.FloatList(value=x["masked_lm_weights"])),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def _serialize_gru4rec_tfrecords(x):
    import tensorflow as tf

    feature = {
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=x["input_ids"])),
        "input_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=x["input_mask"])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=x["label"])),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def dict_to_parquet_table(data):
    import pyarrow as pa
    import pandas as pd

    return pa.Table.from_pandas(pd.DataFrame(data))


def _save_in_parquet(data_dir: str, dataset_dir_version_name: str, examples: PCollection, data_desc: str):
    output_dir = f"{data_dir}/parquets/{dataset_dir_version_name}/{data_desc}"
    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)
    prefix = f"{output_dir}/data"
    schema = pa.schema([
        ("input_ids", pa.list_(pa.int64())),
        ("input_mask", pa.list_(pa.int64())),
        ("label", pa.list_(pa.int64()))
    ])
    examples | f"Write {data_desc} examples" >> beam.io.WriteToParquet(
        prefix,
        schema=schema,
        file_name_suffix=".parquet",
    )


def _save_train_movie_counts(data_dir: str, counts: PCollection):
    counts | "Write train movie counts" >> beam.io.WriteToText(f"{data_dir}/vocab/train_movie_counts.txt", num_shards=1)


def _transform_to_rating(csv_row):
    cells = csv_row.split(",")
    return {"userId": int(cells[0]), "movieId": int(cells[1]), "rating": float(cells[2]), "timestamp": int(cells[3])}


def _filter_examples_per_sample_type(examples_per_user: PCollection, sample_type: int) -> PCollection:
    return examples_per_user | f"Filter {sample_type}" >> beam.Filter(lambda x: x["sample_type"] == sample_type)


def preprocess_with_dataflow(
    data_dir: str,
    dataset_dir_version_name: str,
    max_context_len: int,
    implicit_rating_threshold: float,
    prepare_train_sample_fn,
    prepare_val_test_sample_fn,
    serialize_records_fn,
    duplication_factor: Optional[int] = None,
    nb_max_masked_ids_per_seq: Optional[int] = None,
    mask_ratio: Optional[float] = None,
    proportion_sliding_window: Optional[float] = None,
    sliding_window_step_size_override: Optional[int] = None,
):
    """
    Preprocess the data: read ratings from CSV file and transform them into ready to train serialized tensors in
    Tensorflow tensor records format.
    :param data_dir: The directory from where to find the ratings CSV file
    :param dataset_dir_version_name: The name of the directory under which we will save the tfrecords. Used to identify
    the version of the dataset.
    :param max_context_len: The maximum length a user sequence can have
    :param duplication_factor: The number of time each sequence should be duplicated (considering different random input
    will be masked)
    :param nb_max_masked_ids_per_seq: The maximal number of ids that can be masked in a sequence
    :param mask_ratio: The ratio of input ids to mask for prediction
    :param implicit_rating_threshold: The threshold used to decide whether a rating is an implicit positive sample or
    negative
    :param proportion_sliding_window: The proportion of the complete user sequence that will be used as the size of
    the step in the sliding window used to create the training sequences.
    :param sliding_window_step_size_override: Ability to override the step size proportion with an explicit step size
    number.
    """
    options = beam.pipeline.PipelineOptions(
        runner="DataflowRunner",
        experiments=["use_runner_v2"],
        project="concise-haven-277809",
        staging_location="gs://movie-lens-25m/beam/stg",
        temp_location="gs://movie-lens-25m/beam/tmp",
        job_name="ml-25m-preprocess",
        num_workers=4,
        region="northamerica-northeast2",
        sdk_container_image="northamerica-northeast1-docker.pkg.dev/concise-haven-277809/biarnes-registry/mol-gru4rec-preprocess",
        flink_version="1.16",
    )
    with beam.Pipeline(options=options) as pipeline:
        raw_ratings = (
            pipeline
            | "Read ratings CSV" >> beam.io.textio.ReadFromText(f"{data_dir}/ml-1m/ratings.csv", skip_header_lines=0)
            | "Transform row to rating dict" >> beam.Map(_transform_to_rating)
            | "Filter low ratings (keep implicit positives)" >> beam.Filter(lambda x: x["rating"] > implicit_rating_threshold)
        )

        user_complete_sequences = (
            raw_ratings
            | "Select columns" >> beam.Map(lambda x: (x["userId"], (x["movieId"], x["timestamp"])))
            | "Group By User Id" >> beam.GroupByKey()
            | "Filter If Not Enough Views" >> beam.Filter(lambda x: len(x[1]) >= 5)
            | "Sort Views By Timestamp" >> beam.Map(_sort_views_by_timestamp)
        )

        examples = (
            user_complete_sequences
            | "Generate examples from complete sequences" >>
            beam.Map(
                _generate_examples_from_complete_sequences,
                max_context_len=max_context_len,
                proportion_sliding_window=proportion_sliding_window,
                sliding_window_step_size_override=sliding_window_step_size_override
            )
            | f"Flatten examples" >> beam.FlatMap(lambda x: x)
        )

        # Split examples
        train_examples = _filter_examples_per_sample_type(examples, SampleType.TRAIN)
        val_examples = _filter_examples_per_sample_type(examples, SampleType.VALIDATION)
        test_examples = _filter_examples_per_sample_type(examples, SampleType.TEST)

        # Add masks and augment training data
        train_examples = (
            train_examples
            | "Augment training data and set masks" >> beam.Map(
                prepare_train_sample_fn,
                max_context_len=max_context_len,
                duplication_factor=duplication_factor,
                nb_max_masked_ids_per_seq=nb_max_masked_ids_per_seq,
                mask_ratio=mask_ratio,
            )
            | "Flatten training examples" >> beam.FlatMap(lambda x: x)
        )
        val_examples = (
            val_examples | "Set mask last position - val" >> beam.Map(
                prepare_val_test_sample_fn,
                max_context_len=max_context_len
            )
            | "Flatten val examples" >> beam.FlatMap(lambda x: x)
        )
        test_examples = (
            test_examples | "Set mask last position - test" >> beam.Map(
                prepare_val_test_sample_fn,
                max_context_len=max_context_len
            )
            | "Flatten test examples" >> beam.FlatMap(lambda x: x)
        )

        # Serialize
        train_tf_examples = train_examples | "Serialize train" >> beam.Map(serialize_records_fn) | beam.Reshuffle()
        val_tf_examples = val_examples | "Serialize val" >> beam.Map(serialize_records_fn)
        test_tf_examples = test_examples | "Serialize test" >> beam.Map(serialize_records_fn)

        # Save to disk
        _save_in_parquet(data_dir, dataset_dir_version_name, train_tf_examples, data_desc="train")
        _save_in_parquet(data_dir, dataset_dir_version_name, val_tf_examples, data_desc="val")
        _save_in_parquet(data_dir, dataset_dir_version_name, test_tf_examples, data_desc="test")

        # Count vocab
        train_movie_counts = _count_movies_in_ratings(train_examples)
        _save_train_movie_counts(data_dir, train_movie_counts)


if __name__ == "__main__":
    # BERT4REC preparation
    # preprocess_with_dataflow(
    #     data_dir="gs://movie-lens-25m",
    #     dataset_dir_version_name="bert4rec_ml1m_max40_dup10_05_slidprop",
    #     max_context_len=200,
    #     implicit_rating_threshold=2.0,
    #     prepare_train_sample_fn=_prepare_bert4rec_train_samples,
    #     prepare_val_test_sample_fn=_prepare_bert4rec_val_test_sample,
    #     serialize_records_fn=_serialize_bert4rec_tfrecords,
    #     duplication_factor=10,
    #     nb_max_masked_ids_per_seq=40,
    #     mask_ratio=0.2,
    #     proportion_sliding_window=0.5,
    #     # sliding_window_step_size_override=1
    # )

    # GRU4REC preparation
    preprocess_with_dataflow(
        data_dir="gs://movie-lens-25m",
        dataset_dir_version_name="gru4rec_ml1m_full_slide",
        max_context_len=200,
        implicit_rating_threshold=2.0,
        prepare_train_sample_fn=_prepare_gru4rec_samples,
        prepare_val_test_sample_fn=_prepare_gru4rec_samples,
        serialize_records_fn=dict_to_parquet_table,
        sliding_window_step_size_override=1
    )
