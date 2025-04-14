#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
# Open new terminal
