#!/bin/bash

cd mol-gru4rec/
mkdir -p data/parquets
gsutil -m cp -r gs://movie-lens-25m/ml-1m data/
gsutil -m cp -r gs://movie-lens-25m/vocab data/
gsutil -m cp -r gs://movie-lens-25m/parquets/gru4rec_ml1m_full_slide data/parquets/
uv sync
source .venv/bin/activate
python -m train --data_dir data --config_name gru4rec_config_ml-1m_64_mol_uid_semantic_emb --dataset_name gru4rec_ml1m_full_slide
