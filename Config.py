# -*- coding: utf-8 -*-
import os
import time

import ml_collections
import torch

from paper_experiments import get_paper_experiment


paper_experiment = get_paper_experiment(
    os.environ.get('BETTERLVIT_EXPERIMENT', 'b0_baseline')
)

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = int(os.environ.get('BETTERLVIT_SEED', '1219'))
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # Use cosineLR or not
n_channels = 3
n_labels = 1  # MoNuSeg & Covid19
epochs = int(os.environ.get('BETTERLVIT_EPOCHS', '100'))
img_size = 224
print_frequency = 20
tensorboard_frequency = 20
save_frequency = 5000
# Validation previews are optional training artifacts. Long remote runs can
# disable them without changing optimization or metrics, which also avoids a
# failed image write interrupting checkpoint progress.
vis_frequency = int(os.environ.get('BETTERLVIT_VIS_FREQUENCY', '10'))
early_stopping_patience = 80
print_loss_components = False  # Toggle to print individual loss components

pretrain = False
# task_name = 'MoNuSeg'
task_name = 'Covid19'
learning_rate = 3e-4  # MoNuSeg: 1e-3, Covid19: 3e-4
weight_decay = 1e-4  # L2 regularization on Adam; 0 disables
# The 4090D paper protocol uses one locked physical batch size for B0--A3.
# Every launcher records an explicit override; 16 is the tested server default.
batch_size = int(os.environ.get('BETTERLVIT_BATCH_SIZE', '16'))
train_drop_last = bool(int(os.environ.get('BETTERLVIT_TRAIN_DROP_LAST', '1')))
num_workers = int(os.environ.get('BETTERLVIT_NUM_WORKERS', '4'))
# Recreate workers at every epoch boundary so checkpoint resume can restore the
# exact sampler and augmentation RNG streams.
persistent_workers = False
# Deterministic execution is mandatory for the server paper protocol.
deterministic_training = bool(
    int(os.environ.get('BETTERLVIT_DETERMINISTIC', '1'))
)
cudnn_enabled = bool(int(os.environ.get('BETTERLVIT_CUDNN_ENABLED', '1')))
# Backward-compatible alias used by the existing training entry point.
miopen_enabled = cudnn_enabled

# Pre-registered paper ablation. Boundary supervision is prohibited in every
# profile; only LoRA, objective and decoder fusion are allowed to differ.
boundary_loss_weight = 0.0
boundary_kernel_size = 3
loss_name = paper_experiment['loss_name']
dice_loss_weight = 0.5
focal_loss_weight = 0.5
focal_gamma = 2.0
focal_positive_weight = 0.5
focal_negative_weight = 0.5
experiment_name = paper_experiment['name']
experiment_paper_id = paper_experiment['paper_id']
decoder_fusion_mode = paper_experiment['decoder_fusion_mode']
experiment_architecture = paper_experiment['description']
experiment_architecture_version = paper_experiment['architecture_version']
experiment_output_name = experiment_name + '_evaluation.json'
source_git_commit = os.environ.get('BETTERLVIT_GIT_COMMIT', '').strip()

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
resume_path = os.environ.get('BETTERLVIT_RESUME_PATH', '').strip()
resume_max_dice = 0.0
require_checkpoint_architecture_match = True

# Text encoder (replaces legacy bert-embedding / bert-base-uncased)
text_encoder_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
text_max_len = 32  # threaded into Vit.CTBN3.in_channels via LViT __init__
text_use_lora = paper_experiment['text_use_lora']
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
session_name = (
    experiment_paper_id + '_Test_session' + '_'
    + time.strftime('%m.%d_%Hh%M')
)
save_path = (
    task_name + '/' + model_name + '/' + experiment_name + '/'
    + session_name + '/'
)
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
    config.decoder_fusion_mode = decoder_fusion_mode
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
