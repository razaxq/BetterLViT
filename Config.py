# -*- coding: utf-8 -*-
import os
import re
import time

import ml_collections
import torch

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = int(os.environ.get('BETTERLVIT_SEED', '1219'))
# These streams must stay independent so adding a model branch cannot change
# sample order or augmentation randomness in a paired experiment.
model_seed = seed
training_seed = seed + 100_003
sampler_seed = seed + 200_003
worker_seed = seed + 300_007
validation_worker_seed = seed + 400_009
text_modality_dropout_seed = seed + 500_009
strict_reproducibility = True
git_commit = os.environ.get('BETTERLVIT_GIT_COMMIT', '').strip()
if git_commit and not re.fullmatch(r'[0-9a-f]{40}', git_commit):
    raise ValueError('BETTERLVIT_GIT_COMMIT must be a lowercase 40-hex hash')

cosineLR = True  # Use cosineLR or not
n_channels = 3
n_labels = 1  # MoNuSeg & Covid19
epochs = int(os.environ.get('BETTERLVIT_EPOCHS', '200'))
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
batch_size = int(os.environ.get('BETTERLVIT_BATCH_SIZE', '16'))
num_workers = int(os.environ.get('BETTERLVIT_NUM_WORKERS', '4'))
# Restarting workers at every epoch lets an epoch-boundary checkpoint restore
# their random streams exactly. Persistent worker RNG state is not serializable.
persistent_workers = False

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
resume_path = os.environ.get('BETTERLVIT_RESUME_PATH', '').strip()
resume_sha256 = os.environ.get('BETTERLVIT_RESUME_SHA256', '').strip().lower()
if resume_path and not re.fullmatch(r'[0-9a-f]{64}', resume_sha256):
    raise ValueError(
        'BETTERLVIT_RESUME_SHA256 is required with BETTERLVIT_RESUME_PATH'
    )
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

# Pre-registered next-stage variable. The grouped V4-B rebaseline uses 0.0;
# its paired candidate uses exactly 0.5 with every other setting unchanged.
text_modality_dropout_prob = float(os.environ.get(
    'BETTERLVIT_TEXT_MODALITY_DROPOUT_PROB',
    '0.0',
))
if text_modality_dropout_prob not in (0.0, 0.5):
    raise ValueError(
        'BETTERLVIT_TEXT_MODALITY_DROPOUT_PROB must be 0.0 or 0.5'
    )
text_modality_dropout_prompt = 'No report available.'

dataset_root = './datasets/' + task_name + '/'
train_dataset = dataset_root + 'Train_Folder/'
val_dataset = dataset_root + 'Val_Folder/'
test_dataset = dataset_root + 'Test_Folder/'
task_dataset = train_dataset

# ``legacy`` preserves the published LViT image-level split. The grouped
# protocol uses a committed manifest generated only from the original 7,145
# train+validation pool; the official test folder remains locked.
split_protocol = os.environ.get(
    'BETTERLVIT_SPLIT_PROTOCOL',
    'legacy',
).strip()
split_manifest_path = os.environ.get(
    'BETTERLVIT_SPLIT_MANIFEST',
    '',
).strip()
allowed_split_protocols = {
    'legacy',
    'known_patient_grouped_sensitivity_v1',
}
if split_protocol not in allowed_split_protocols:
    raise ValueError('Unsupported BETTERLVIT_SPLIT_PROTOCOL: {!r}'.format(
        split_protocol
    ))
if split_protocol != 'legacy' and not split_manifest_path:
    raise ValueError(
        'BETTERLVIT_SPLIT_MANIFEST is required for grouped training'
    )

session_name = os.environ.get(
    'BETTERLVIT_SESSION_NAME',
    'Test_session_' + time.strftime('%Y%m%d_%H%M%S'),
).strip()
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', session_name):
    raise ValueError('Unsafe BETTERLVIT_SESSION_NAME: {!r}'.format(
        session_name
    ))
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
