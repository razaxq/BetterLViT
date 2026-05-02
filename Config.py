# -*- coding: utf-8 -*-
import ml_collections
import os
import time
import torch

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 1219
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # Use cosineLR or not
n_channels = 3
n_labels = 1  # MoNuSeg & Covid19
epochs = 200
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 80
print_loss_components = False  # Toggle to print individual loss components

pretrain = False
# task_name = 'MoNuSeg'
task_name = 'Covid19'
learning_rate = 3e-4  # MoNuSeg: 1e-3, Covid19: 3e-4
weight_decay = 1e-4  # L2 regularization on Adam; 0 disables
batch_size = 16  # For LViT-T, 2 is better than 4

model_name = 'BetterLViT'
# model_name = 'LViT_pretrain'

# Resume training
# Set resume_path to a .pth.tar checkpoint to continue from there. New session
# (and its log / checkpoint folder) is still created on each run, so the
# original best_model is not overwritten in the source session.
# resume_max_dice is only used as a fallback when the loaded checkpoint
# predates this resume infrastructure (no 'max_dice' field).
resume_path = ''
resume_max_dice = 0.0

# Text encoder (replaces legacy bert-embedding / bert-base-uncased)
text_encoder_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
text_max_len = 32  # tokenizer max_length; cross-attn handles arbitrary seq_len
text_use_lora = True
text_lora_r = 8
text_lora_alpha = 16
text_lora_dropout = 0.15

train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
task_dataset = './datasets/' + task_name + '/Train_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config


test_session = ""
