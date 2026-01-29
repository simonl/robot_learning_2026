# !/bin/bash

# Install packages into virtual environment
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -U wandb