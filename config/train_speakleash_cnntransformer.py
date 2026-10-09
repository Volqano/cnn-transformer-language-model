# config for a first Speakleash training run with CNN injected before the Transformer
# launch with:
#   python train.py config/train_speakleash.py

# WandB
wandb_log = False
wandb_project = 'speakleash'
wandb_run_name = 'bielik-cnn-transformer'

# dataset and output
dataset = 'speakleash'
out_dir = 'out-speakleash-cnn'

# Model
block_size = 1024
batch_size = 12
gradient_accumulation_steps = 5
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# CNN
use_cnn = True
cnn_kernel_size = 3

# Training
learning_rate = 6e-4
max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 2000
min_lr = 6e-5

# Evaluation and logging
eval_interval = 200
eval_iters = 50
log_interval = 10
always_save_checkpoint = True

# System
device = 'cuda'
compile = False
