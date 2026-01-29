# !/bin/bash

# Setup script for robot_learning_2026 environment

# Install packages into virtual environment
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -U wandb

# Update .bashrc to activate virtual environment on login
echo 'cd /root/robot_learning_2026/' >> ~/.bashrc
echo 'source .venv/bin/activate' >> ~/.bashrc
echo './activate.sh' >> ~/.bashrc

# Configure git user information
git config --global user.name "Simon Langlois"
git config --global user.email "simon.langlois.4@umontreal.ca"
