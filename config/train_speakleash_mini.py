# WandB
wandb_log = False
wandb_project = 'speakleash'
wandb_run_name = 'bielik-cnn-transformer-mini'

# Data
dataset = 'speakleash'
out_dir = 'out-sanity-cnn'

# Model
block_size = 256
batch_size = 4
gradient_accumulation_steps = 4
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False

# CNN
use_cnn = True
cnn_kernel_size = 3

# Training
learning_rate = 3e-4
max_iters = 100
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 10
lr_decay_iters = 100
min_lr = 3e-5

# Evaluation and logging
eval_interval = 25
eval_iters = 10
log_interval = 10
always_save_checkpoint = True

# System
device = 'cuda'
compile = False