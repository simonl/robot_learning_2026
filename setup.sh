# !/bin/bash

# Setup script for robot_learning_2026 environment

# Install Python 3.10 
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install packages into virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -U wandb

# Update .bashrc to activate virtual environment on login
echo 'cd /root/robot_learning_2026/' >> ~/.bashrc
echo './activate.sh' >> ~/.bashrc

# Configure git user information
git config --global user.name "Simon Langlois"
git config --global user.email "simon.langlois.4@umontreal.ca"
