import glob
import tensorflow as tf
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def _get_directory(data_dir: str, format_type: str, dataset_name: str, dataset_type: str):
    return f"{data_dir}/{format_type}/{dataset_name}/{dataset_type}"


def _get_tfrecords_file_list(input_dir: str):
    return


def convert_tfrecords_to_parquet(data_dir: str, dataset_name: str, dataset_type: str):
    input_dir = _get_directory(data_dir, "tfrecords", dataset_name, dataset_type)
    output_dir = _get_directory(data_dir, "parquets", dataset_name, dataset_type)
    tfrecords_file_list =  glob.glob(f"{input_dir}/data-00022-*.gz")
    for tfrecord_file in tfrecords_file_list:
        # Read the TFRecord file
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        records = [record.numpy() for record in dataset]

        # Convert to Parquet
        table = pa.Table.from_pandas(pd.DataFrame(records))
        pq.write_table(table, f"{output_dir}/{tfrecord_file.split('/')[-1].replace('.gz', '.parquet')}")


if __name__ == "__main__":
    convert_tfrecords_to_parquet("data", "gru4rec_full_slide", "train")
