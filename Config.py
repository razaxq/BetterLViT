# -*- coding: utf-8 -*-
import os
import time

import ml_collections
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
num_workers = 4
persistent_workers = True

# FAM-EPPA V4-B architecture ablation. V4-A remains intact, while up4/up3 add
# efficient spatially adaptive low/high-pass routing. Boundary supervision
# remains strictly disabled so this stays an architecture-only experiment.
boundary_loss_weight = 0.0
boundary_kernel_size = 3
loss_name = 'dice_focal'
dice_loss_weight = 0.5
focal_loss_weight = 0.5
focal_gamma = 2.0
focal_positive_weight = 0.5
focal_negative_weight = 0.5
experiment_architecture = 'FAM-EPPA V4-B (Low-Resolution Adaptive ALPF/AHPF)'
experiment_architecture_version = 'fam_eppa_v4b'
experiment_output_name = 'fam_eppa_v4b_evaluation.json'

model_name = 'BetterLViT'
# model_name = 'LViT_pretrain'

# Local workstation safety.
enable_bark_notifications = False
shutdown_after_training = False

# Resume training
# Set resume_path to a .pth.tar checkpoint to continue from there. New session
# (and its log / checkpoint folder) is still created on each run, so the
# original best_model is not overwritten in the source session.
# resume_max_dice is only used as a fallback when the loaded checkpoint
# predates this resume infrastructure (no 'max_dice' field).
resume_path = ''
resume_max_dice = 0.0
require_checkpoint_architecture_match = True

# Text encoder (replaces legacy bert-embedding / bert-base-uncased)
text_encoder_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
text_max_len = 32  # threaded into Vit.CTBN3.in_channels via LViT __init__
text_use_lora = True
text_lora_r = 16
text_lora_alpha = 32
text_lora_dropout = 0.1
# LoRA target modules. PEFT does suffix matching, so 'output.dense' matches
# BOTH attention.output.dense (attention "o" projection) AND the FFN
# output.dense (3072->768). Default below covers all 6 linears per BERT
# block: query, key, value, attention.output.dense, intermediate.dense
# (FFN up 768->3072), output.dense (FFN down). Reduce to ('query', 'value')
# for the legacy q+v-only ablation comparison.
text_lora_target_modules = (
    'query', 'key', 'value',
    'intermediate.dense', 'output.dense',
)

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
    # FAM-EPPA V4-B structural switches and residual bounds.
    config.eppa_use_decoder_guide = True
    config.eppa_use_dilated_edge = True
    config.eppa_use_text_pixel_film = True
    config.eppa_use_plam_guide = True
    config.eppa_normalize_channel_descriptors = True
    config.eppa_channel_strength_max = 0.5
    config.eppa_pixel_strength_max = 0.35
    config.eppa_edge_strength_max = 0.30
    config.eppa_plam_strength_max = 1.25
    config.eppa_plam_strength_init = 1.0
    config.eppa_plam_strength_floor = 0.25
    config.eppa_detail_strength_floor = 0.02
    # Keep the ablation localized: only the two lowest-resolution decoder
    # stages receive adaptive frequency filtering.
    config.eppa_adaptive_frequency_stages = ('up4', 'up3')
    config.eppa_frequency_groups = 8
    config.eppa_frequency_context_channels = 32
    config.eppa_alpf_strength_max = 0.50
    config.eppa_alpf_strength_init = 0.20
    config.eppa_ahpf_strength_max = 0.30
    config.eppa_ahpf_strength_init = 0.08
    config.eppa_ahpf_strength_floor = 0.02
    return config


test_session = ""
