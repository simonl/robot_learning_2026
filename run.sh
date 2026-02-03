# !/bin/bash

# Run script for robot_learning_2026 environment
#python mini-grp/mini_grp.py dataset.buffer_size=1000 trim=1000 n_embd=256 batch_size=64 dataset.encode_with_t5=false data_shuffel_interval=10 eval_interval=10 dataset.num_episodes=1 dataset.chunk_size=1

python mini-grp/mini_grp.py dataset.buffer_size=256 n_embd=128 batch_size=64 dataset.encode_with_t5=false data_shuffel_interval=10 eval_interval=10 dataset.chunk_size=1
