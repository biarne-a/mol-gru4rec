#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/biarne-a/mol-gru4rec.git
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
# Open new terminal

gcloud auth login --update-adc
gcloud config set project concise-haven-277809

cd mol-gru4rec/
mkdir -p data/parquets
gsutil -m cp -r gs://movie-lens-25m/ml-1m data/
gsutil -m cp -r gs://movie-lens-25m/vocab data/
gsutil -m cp -r gs://movie-lens-25m/parquets/gru4rec_ml1m_full_slide data/parquets/
uv sync
source .venv/bin/activate
python -m train --data_dir data --config_name gru4rec_config_ml-1m_64_mol_uid --dataset_name gru4rec_ml1m_full_slide
