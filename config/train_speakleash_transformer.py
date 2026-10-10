# config for a first Speakleash training run with CNN injected before the Transformer
# launch with:
#   python train.py config/train_speakleash.py

# WandB
wandb_log = True
wandb_project = 'polish-cnn-transformer'
wandb_run_name = 'transformer-baseline-5000'

# dataset and output
dataset = 'speakleash'
out_dir = 'out-speakleash-transformer-5000'

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
use_cnn = False
cnn_kernel_size = 3

# Training
learning_rate = 6e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

# Evaluation and logging
eval_interval = 250
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

# System
device = 'cuda'
compile = False