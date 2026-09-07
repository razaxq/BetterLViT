# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import subprocess

import requests
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

import Config as config
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
from Train_one_epoch import train_one_epoch
from nets.BetterLViT import BetterLViT
from reproducibility import (
    PROTOCOL_VERSION,
    build_run_fingerprint,
    capture_rng_state,
    configure_determinism,
    make_generator,
    named_file_set_sha256,
    reset_global_seed,
    restore_rng_state,
    seed_worker,
    sha256_file,
    validate_resume_fingerprint,
)
from split_protocol import load_grouped_manifest
from utils import (
    CosineAnnealingWarmRestarts,
    WeightedDiceBCE,
    WeightedDiceFocal,
    read_text,
)


def bark_notify(body, title="训练通知"):
    """极简版：只发送标题和文字内容"""
    if not getattr(config, 'enable_bark_notifications', False):
        return
    bark_key = "uAnJRvt7pxbzE9KK6bCVva"
    url = f"https://api.day.app/{bark_key}/{title}/{body}"
    try:
        # 短 timeout: 网络不可达时直接放过，避免训练脚本被 Bark 阻塞
        requests.get(url, timeout=3)
    except Exception as e:
        print(f"推送失败: {e}")

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def verify_source_checkout(expected_commit):
    repository = Path(__file__).resolve().parent
    try:
        actual_commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain=v1'],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Unable to verify source Git checkout') from exc
    if actual_commit != expected_commit:
        raise RuntimeError(
            'Source commit mismatch: {} != {}'.format(
                actual_commit,
                expected_commit,
            )
        )
    if dirty:
        raise RuntimeError(
            'Refusing to train from a dirty worktree:\n{}'.format(dirty)
        )


def save_checkpoint(state, save_path, verbose=True):
    '''
        Save model checkpoint. best_model=True writes best_model-{model}.pth.tar;
        otherwise writes last_model-{model}.pth.tar (rolling, overwrites each call).
    '''
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + 'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + 'last_model-{}.pth.tar'.format(model)
    if verbose:
        logger.info('\t Saving to {}'.format(filename))
    temporary = filename + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, filename)


def build_checkpoint_state(
    model,
    optimizer,
    lr_scheduler,
    model_type,
    epoch,
    val_loss,
    max_dice,
    best_epoch,
    epoch_history,
    is_best,
    run_fingerprint,
    rng_generators,
):
    return {
        'epoch': epoch,
        'best_model': is_best,
        'model': model_type,
        'state_dict': model.state_dict(),
        'val_loss': val_loss,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'max_dice': float(max_dice),
        'best_epoch': int(best_epoch),
        'epoch_history': epoch_history,
        'architecture': getattr(
            config,
            'experiment_architecture',
            None,
        ),
        'architecture_version': getattr(
            config,
            'experiment_architecture_version',
            None,
        ),
        'reproducibility_protocol': PROTOCOL_VERSION,
        'text_modality_dropout_prob': config.text_modality_dropout_prob,
        'text_modality_dropout_prompt': config.text_modality_dropout_prompt,
        'run_fingerprint': run_fingerprint,
        'rng_state': capture_rng_state(rng_generators),
    }


def compute_eppa_stats(model):
    """Snapshot the latest validation-time EPPA statistics."""
    target = model.module if isinstance(model, nn.DataParallel) else model
    stats = {}
    for stage in ('up4', 'up3', 'up2', 'up1'):
        block = getattr(target, stage, None)
        if block is None or not hasattr(block, 'eppa'):
            continue
        stage_stats = getattr(block.eppa, '_last_stats', None)
        if not stage_stats:
            continue
        stats[stage] = dict(stage_stats)
    return stats


def build_optimizer_parameter_groups(model, weight_decay):
    """Keep residual gates and normalization parameters free of L2 decay."""
    decay_parameters = []
    no_decay_parameters = []
    decay_names = []
    no_decay_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized_name = name.lower()
        use_no_decay = (
            parameter.ndim <= 1
            or name.endswith('.bias')
            or 'strength_logit' in normalized_name
            or 'norm' in normalized_name
        )
        if use_no_decay:
            no_decay_parameters.append(parameter)
            no_decay_names.append(name)
        else:
            decay_parameters.append(parameter)
            decay_names.append(name)
    parameter_groups = []
    if decay_parameters:
        parameter_groups.append({
            'params': decay_parameters,
            'weight_decay': weight_decay,
        })
    if no_decay_parameters:
        parameter_groups.append({
            'params': no_decay_parameters,
            'weight_decay': 0.0,
        })
    return parameter_groups, decay_names, no_decay_names


def _load_datasets(train_tf, val_tf):
    split_context = {
        'manifest_path': '',
        'manifest_sha256': '',
        'text_workbook_paths': [],
    }
    if config.task_name == 'MoNuSeg':
        if config.split_protocol != 'legacy':
            raise RuntimeError('Grouped protocol is implemented only for Covid19')
        train_text_path = config.train_dataset + 'Train_text.xlsx'
        val_text_path = config.val_dataset + 'Val_text.xlsx'
        train_text = read_text(train_text_path)
        val_text = read_text(val_text_path)
        split_context['text_workbook_paths'] = [
            ('train_text', train_text_path),
            ('validation_text', val_text_path),
        ]
        train_dataset = ImageToImage2D(
            config.train_dataset,
            config.task_name,
            train_text,
            train_tf,
            image_size=config.img_size,
        )
        val_dataset = ImageToImage2D(
            config.val_dataset,
            config.task_name,
            val_text,
            val_tf,
            image_size=config.img_size,
        )
        return train_dataset, val_dataset, split_context

    if config.task_name != 'Covid19':
        raise RuntimeError('Unsupported task: {}'.format(config.task_name))
    text_workbook_path = config.task_dataset + 'Train_Val_text.xlsx'
    text = read_text(text_workbook_path)
    split_context['text_workbook_paths'] = [
        ('train_validation_text', text_workbook_path),
    ]
    train_records = None
    validation_records = None
    if config.split_protocol != 'legacy':
        grouped = load_grouped_manifest(
            config.split_manifest_path,
            config.dataset_root,
            config.split_protocol,
        )
        train_records = grouped['train_records']
        validation_records = grouped['validation_records']
        split_context.update({
            'manifest_path': grouped['manifest_path'],
            'manifest_sha256': grouped['manifest_sha256'],
        })
    train_dataset = ImageToImage2D(
        config.train_dataset,
        config.task_name,
        text,
        train_tf,
        image_size=config.img_size,
        sample_records=train_records,
    )
    val_dataset = ImageToImage2D(
        config.val_dataset,
        config.task_name,
        text,
        val_tf,
        image_size=config.img_size,
        sample_records=validation_records,
    )
    return train_dataset, val_dataset, split_context


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    train_dataset, val_dataset, split_context = _load_datasets(
        train_tf,
        val_tf,
    )
    dataset_artifact_paths = []
    for split_name, dataset in (
        ('train', train_dataset),
        ('validation', val_dataset),
    ):
        for record in dataset.sample_records:
            image_name = record['image_name']
            dataset_artifact_paths.extend((
                (
                    '{}/image/{}'.format(split_name, image_name),
                    record['image_path'],
                ),
                (
                    '{}/mask/{}'.format(split_name, record['mask_name']),
                    record['mask_path'],
                ),
            ))
    dataset_artifacts_sha256 = named_file_set_sha256(
        dataset_artifact_paths
    )
    text_workbook_sha256 = named_file_set_sha256(
        split_context['text_workbook_paths']
    )
    rng_generators = {
        'train_sampler': make_generator(config.sampler_seed),
        'train_workers': make_generator(config.worker_seed),
        'validation_workers': make_generator(config.validation_worker_seed),
        'text_modality_dropout': make_generator(
            config.text_modality_dropout_seed
        ),
    }
    text_dropout_context = {
        'probability': config.text_modality_dropout_prob,
        'generator': rng_generators['text_modality_dropout'],
        'neutral_input_ids': train_dataset.neutral_input_ids,
        'neutral_attention_mask': train_dataset.neutral_attention_mask,
    }
    train_sampler = RandomSampler(
        train_dataset,
        generator=rng_generators['train_sampler'],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        worker_init_fn=seed_worker,
        generator=rng_generators['train_workers'],
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(
            config.persistent_workers and config.num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=rng_generators['validation_workers'],
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(
            config.persistent_workers and config.num_workers > 0
        ),
    )
    run_fingerprint = build_run_fingerprint(
        seed=config.seed,
        split_protocol=config.split_protocol,
        split_manifest_path=split_context['manifest_path'],
        split_manifest_sha256=split_context['manifest_sha256'],
        train_names=train_dataset.images_list,
        validation_names=val_dataset.images_list,
        text_modality_dropout_prob=config.text_modality_dropout_prob,
        text_modality_dropout_prompt=config.text_modality_dropout_prompt,
        git_commit=config.git_commit,
        dataset_artifacts_sha256=dataset_artifacts_sha256,
        text_workbook_sha256=text_workbook_sha256,
        training_configuration={
            'architecture_version': config.experiment_architecture_version,
            'model_name': config.model_name,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'num_workers': config.num_workers,
            'persistent_workers': config.persistent_workers,
            'image_size': config.img_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'cosine_lr': config.cosineLR,
            'early_stopping_patience': config.early_stopping_patience,
            'loss_name': config.loss_name,
            'dice_loss_weight': config.dice_loss_weight,
            'focal_loss_weight': config.focal_loss_weight,
            'focal_gamma': config.focal_gamma,
            'focal_positive_weight': config.focal_positive_weight,
            'focal_negative_weight': config.focal_negative_weight,
            'boundary_loss_weight': config.boundary_loss_weight,
            'text_encoder_name': config.text_encoder_name,
            'text_max_len': config.text_max_len,
            'text_use_lora': config.text_use_lora,
            'text_lora_r': config.text_lora_r,
            'text_lora_alpha': config.text_lora_alpha,
            'text_lora_dropout': config.text_lora_dropout,
            'text_lora_target_modules': config.text_lora_target_modules,
            'model_seed': config.model_seed,
            'training_seed': config.training_seed,
            'sampler_seed': config.sampler_seed,
            'worker_seed': config.worker_seed,
            'validation_worker_seed': config.validation_worker_seed,
            'text_modality_dropout_seed': (
                config.text_modality_dropout_seed
            ),
        },
    )
    logger.info('Reproducibility fingerprint: {}'.format(
        run_fingerprint['fingerprint_sha256']
    ))
    logger.info(
        'Data artifacts: samples_sha256={}, text_workbook_sha256={}'.format(
            dataset_artifacts_sha256,
            text_workbook_sha256,
        )
    )
    logger.info(
        'Split protocol: {} (train={}, validation={}, manifest_sha256={})'.format(
            config.split_protocol,
            len(train_dataset),
            len(val_dataset),
            split_context['manifest_sha256'] or 'none',
        )
    )
                             
    lr = config.learning_rate
    logger.info(model_type)

    if model_type in ('LViT', 'BetterLViT'):
        reset_global_seed(config.model_seed)
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        # 'LViT' = frozen CXR-BERT baseline, 'BetterLViT' = LoRA-tuned CXR-BERT
        use_lora = config.text_use_lora and (model_type == 'BetterLViT')
        model = BetterLViT(
            config_vit,
            n_channels=config.n_channels,
            n_classes=config.n_labels,
            text_encoder_name=config.text_encoder_name,
            text_seq_len=config.text_max_len,
            use_lora=use_lora,
            lora_r=config.text_lora_r,
            lora_alpha=config.text_lora_alpha,
            lora_dropout=config.text_lora_dropout,
            lora_target_modules=config.text_lora_target_modules,
        )

    elif model_type == 'LViT_pretrain':
        reset_global_seed(config.model_seed)
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = BetterLViT(
            config_vit,
            n_channels=config.n_channels,
            n_classes=config.n_labels,
            text_encoder_name=config.text_encoder_name,
            text_seq_len=config.text_max_len,
            use_lora=config.text_use_lora,
            lora_r=config.text_lora_r,
            lora_alpha=config.text_lora_alpha,
            lora_dropout=config.text_lora_dropout,
            lora_target_modules=config.text_lora_target_modules,
        )
        pretrained_UNet_model_path = "MoNuSeg/LViT/Test_session_05.23_10h55/models/best_model-LViT.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)
        logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')
    # thop is incompatible with PEFT-wrapped modules (double-registration
    # leaves stale CPU hooks that break training after .cuda()). Report
    # parameter counts directly — FLOPs aren't needed for the LoRA setup.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params: {} ({:.2f}M)'.format(total, total / 1e6))
    print('trainable params: {} ({:.2f}M, {:.2%})'.format(
        trainable, trainable / 1e6, trainable / max(total, 1)))
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    configured_loss = getattr(config, 'loss_name', 'dice_bce')
    if configured_loss == 'dice_focal':
        if config.boundary_loss_weight != 0.0:
            raise ValueError(
                'dice_focal requires boundary_loss_weight=0.0'
            )
        criterion = WeightedDiceFocal(
            dice_weight=config.dice_loss_weight,
            focal_weight=config.focal_loss_weight,
            focal_gamma=config.focal_gamma,
            focal_positive_weight=config.focal_positive_weight,
            focal_negative_weight=config.focal_negative_weight,
        )
    elif configured_loss == 'dice_bce':
        criterion = WeightedDiceBCE(
            dice_weight=0.5,
            BCE_weight=0.5,
            boundary_weight=config.boundary_loss_weight,
            boundary_kernel_size=config.boundary_kernel_size,
        )
    else:
        raise ValueError(
            'Unsupported loss_name: {}'.format(configured_loss)
        )
    logger.info(
        'Objective: {} (boundary_weight={:.1f})'.format(
            configured_loss,
            config.boundary_loss_weight,
        )
    )
    optimizer_groups, decay_names, no_decay_names = (
        build_optimizer_parameter_groups(
            model,
            getattr(config, 'weight_decay', 0.0),
        )
    )
    optimizer = torch.optim.Adam(optimizer_groups, lr=lr)
    logger.info(
        'Optimizer groups: decay={} tensors, no_decay={} tensors; '
        'EPPA residual gates are protected from L2 decay'.format(
            len(decay_names),
            len(no_decay_names),
        )
    )
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # Model construction is intentionally isolated from the operation RNG.
    # Paired candidates therefore start with identical shared weights and
    # identical dropout/random-op streams.
    reset_global_seed(config.training_seed)
    max_dice = 0.0
    best_epoch = 1
    epoch_history = []
    start_epoch = 0

    # ------------------------- Resume from checkpoint -------------------------
    if config.resume_path:
        if os.path.isfile(config.resume_path):
            logger.info('Resuming from {}'.format(config.resume_path))
            actual_resume_sha256 = sha256_file(config.resume_path)
            if actual_resume_sha256 != config.resume_sha256:
                raise RuntimeError(
                    'Resume checkpoint SHA-256 mismatch: {} != {}'.format(
                        actual_resume_sha256,
                        config.resume_sha256,
                    )
                )
            # Resume checkpoints are self-generated trusted artifacts and
            # include Python/NumPy RNG state in addition to tensor weights.
            ckpt = torch.load(
                config.resume_path,
                map_location='cpu',
                weights_only=False,
            )

            expected_architecture = getattr(
                config,
                'experiment_architecture_version',
                None,
            )
            checkpoint_architecture = ckpt.get('architecture_version')
            if (
                getattr(
                    config,
                    'require_checkpoint_architecture_match',
                    False,
                )
                and checkpoint_architecture != expected_architecture
            ):
                raise RuntimeError(
                    'Checkpoint architecture mismatch: expected {!r}, '
                    'found {!r}. This experiment must train from scratch or '
                    'resume from an architecture-matched checkpoint.'.format(
                        expected_architecture,
                        checkpoint_architecture,
                    )
                )

            target = model.module if isinstance(model, nn.DataParallel) else model
            target.load_state_dict(ckpt['state_dict'], strict=True)
            validate_resume_fingerprint(
                ckpt.get('run_fingerprint'),
                run_fingerprint,
            )
            optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = ckpt['epoch'] + 1

            if lr_scheduler is not None:
                if ckpt.get('lr_scheduler') is not None:
                    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                else:
                    raise RuntimeError(
                        'Strict resume requires lr_scheduler state'
                    )

            max_dice = float(ckpt.get('max_dice', config.resume_max_dice))
            best_epoch = int(ckpt.get('best_epoch', start_epoch))
            epoch_history = ckpt.get('epoch_history', []) or []
            if (
                len(epoch_history) != start_epoch
                or not epoch_history
                or int(epoch_history[-1].get('epoch', -1)) != start_epoch
            ):
                raise RuntimeError(
                    'Checkpoint epoch_history is incomplete or inconsistent'
                )
            restore_rng_state(ckpt.get('rng_state'), rng_generators)

            logger.info('Resumed at epoch {}, max_dice={:.4f}, best_epoch={}, history rows={}'.format(
                start_epoch + 1, max_dice, best_epoch, len(epoch_history)))
        else:
            raise FileNotFoundError(
                'resume_path set but file not found: {}'.format(
                    config.resume_path
                )
            )
    # --------------------------------------------------------------------------

    for epoch in range(start_epoch, config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs))
        logger.info(config.session_name)
        # Capture LR used for this epoch (scheduler steps inside the val call, so
        # snapshotting before train gives the actual learning rate this epoch ran on)
        epoch_lr = min(g["lr"] for g in optimizer.param_groups)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_loss, train_dice, train_iou = train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None,
                                                            model_type, logger,
                                                            text_dropout_context=text_dropout_context)  # sup
        train_loss_components = dict(
            getattr(criterion, 'last_epoch_components', {})
        )
        train_text_dropout_stats = dict(
            getattr(criterion, 'last_text_dropout_stats', {})
        )

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice, val_iou = train_one_epoch(val_loader, model, criterion,
                                                          optimizer, writer, epoch, lr_scheduler, model_type, logger)
        val_loss_components = dict(
            getattr(criterion, 'last_epoch_components', {})
        )
        # Append current epoch to history BEFORE saving any checkpoint, so that
        # both best_model and last_model serialise an epoch_history that
        # includes the just-finished epoch (the best_model path used to drop
        # its own row otherwise).
        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_dice': float(train_dice),
            'train_iou': float(train_iou),
            'val_loss': float(val_loss),
            'val_dice': float(val_dice),
            'val_iou': float(val_iou),
            'lr': float(epoch_lr),
            'train_loss_components': train_loss_components,
            'train_text_dropout': train_text_dropout_stats,
            'val_loss_components': val_loss_components,
            'eppa_stats': compute_eppa_stats(model),
        })

        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 5:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                best_state = build_checkpoint_state(
                    model, optimizer, lr_scheduler, model_type, epoch,
                    val_loss, max_dice, best_epoch, epoch_history,
                    is_best=True,
                    run_fingerprint=run_fingerprint,
                    rng_generators=rng_generators,
                )
                save_checkpoint(best_state, config.model_path)
                bark_notify(f"当前最高 Dice 刷新为: {max_dice:.4f}！", title="nb 兄弟")
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        # Always save last_model (rolling) so future runs can resume from any
        # interruption point, not just from the best.
        last_state = build_checkpoint_state(
            model, optimizer, lr_scheduler, model_type, epoch,
            val_loss, max_dice, best_epoch, epoch_history,
            is_best=False,
            run_fingerprint=run_fingerprint,
            rng_generators=rng_generators,
        )
        save_checkpoint(last_state, config.model_path, verbose=False)
        logger.info('--- Epoch History (1..{}) ---'.format(epoch + 1))
        logger.info('{:>5} | {:>10} | {:>10} | {:>9} | {:>10} | {:>10} | {:>9} | {:>10} | {:>4}'.format(
            'Epoch', 'TrainLoss', 'TrainDice', 'TrainIoU', 'ValLoss', 'ValDice', 'ValIoU', 'LR', 'Best'))
        for h in epoch_history:
            marker = '*' if h['epoch'] == best_epoch else ''
            logger.info('{:>5d} | {:>10.4f} | {:>10.4f} | {:>9.4f} | {:>10.4f} | {:>10.4f} | {:>9.4f} | {:>10.2e} | {:>4}'.format(
                h['epoch'], h['train_loss'], h['train_dice'], h['train_iou'],
                h['val_loss'], h['val_dice'], h['val_iou'], h['lr'], marker))

        # Current validation snapshot is persisted in epoch_history while the
        # log stays compact instead of reprinting an O(epoch^2) stats table.
        current_eppa_stats = epoch_history[-1].get('eppa_stats') or {}
        if current_eppa_stats:
            logger.info(
                '--- {} validation statistics ---'.format(
                    getattr(
                        config,
                        'experiment_architecture',
                        'EPPA',
                    )
                )
            )
            for stage in ('up4', 'up3', 'up2', 'up1'):
                stage_stats = current_eppa_stats.get(stage)
                if not stage_stats:
                    continue
                logger.info(
                    '{}: ca={:.4f}+/-{:.4f}, sa={:.4f}+/-{:.4f}, '
                    'amp={:.4f}, suppress={:.4f}, guide_abs={:.4f}'.format(
                        stage,
                        stage_stats['channel_mean'],
                        stage_stats['channel_std'],
                        stage_stats['spatial_mean'],
                        stage_stats['spatial_std'],
                        stage_stats['spatial_amplify_ratio'],
                        stage_stats['spatial_suppress_ratio'],
                        stage_stats['guide_abs_mean'],
                    )
                )
                logger.info(
                    '{} residuals: local_mean={:.3e}, global_mean={:.4f}, '
                    'local_strength={:.4f}, global_strength={:.4f}, '
                    'gain=[{:.4f},{:.4f}], saturation={:.4f}, '
                    'text_film={:.4f}'.format(
                        stage,
                        stage_stats['spatial_local_mean'],
                        stage_stats['spatial_global_mean'],
                        stage_stats['local_strength_mean'],
                        stage_stats['global_strength_mean'],
                        stage_stats['spatial_min'],
                        stage_stats['spatial_max'],
                        stage_stats['spatial_saturation_ratio'],
                        stage_stats['text_film_abs_mean'],
                    )
                )
                if (
                    stage_stats.get('architecture_version')
                    in ('fam_eppa_v4a', 'fam_eppa_v4b')
                ):
                    logger.info(
                        '{} haar: reconstruction_error={:.3e}, '
                        'skip_low/high={:.4f}/{:.4f}, '
                        'plam_low/high={:.4f}/{:.4f}'.format(
                            stage,
                            stage_stats['haar_reconstruction_error'],
                            stage_stats['skip_low_energy_ratio'],
                            stage_stats['skip_high_energy_ratio'],
                            stage_stats['plam_low_energy_ratio'],
                            stage_stats['plam_high_energy_ratio'],
                        )
                    )
                    logger.info(
                        '{} strengths: plam={:.4f}, region={:.4f}, '
                        'detail={:.4f}; residual_std region/detail/refine='
                        '{:.4f}/{:.4f}/{:.4f}'.format(
                            stage,
                            stage_stats['plam_strength_mean'],
                            stage_stats['region_strength_mean'],
                            stage_stats['detail_strength_mean'],
                            stage_stats['region_residual_std'],
                            stage_stats['detail_residual_std'],
                            stage_stats['detail_refinement_std'],
                        )
                    )
                    logger.info(
                        '{} guide_mix: entropy={:.4f}, raw_low={:.4f}, '
                        'plam_low={:.4f}, decoder_low={:.4f}, detail={:.4f}; '
                        'agreement plam/decoder={:.4f}/{:.4f}, '
                        'support={:.4f}+/-{:.4f}'.format(
                            stage,
                            stage_stats['guide_branch_entropy'],
                            stage_stats['guide_skip_weight'],
                            stage_stats['guide_plam_weight'],
                            stage_stats['guide_decoder_weight'],
                            stage_stats['guide_detail_weight'],
                            stage_stats['plam_skip_agreement'],
                            stage_stats['decoder_skip_agreement'],
                            stage_stats['semantic_support_mean'],
                            stage_stats['semantic_support_std'],
                        )
                    )
                    if stage_stats.get('adaptive_frequency_enabled'):
                        logger.info(
                            '{} adaptive_frequency: ALPF strength={:.4f}, '
                            'weights={:.4f}/{:.4f}/{:.4f}, '
                            'sum={:.4f}, entropy={:.4f}, delta_std={:.4f}; '
                            'AHPF strength={:.4f}, '
                            'weights={:.4f}/{:.4f}/{:.4f}, '
                            'sum={:.4f}, entropy={:.4f}, residual_std={:.4f}, '
                            'scaled_std={:.4f}'.format(
                                stage,
                                stage_stats['alpf_strength_mean'],
                                stage_stats['alpf_identity_weight'],
                                stage_stats['alpf_blur3_weight'],
                                stage_stats['alpf_blur5_weight'],
                                stage_stats['alpf_kernel_sum'],
                                stage_stats['alpf_kernel_entropy'],
                                stage_stats['alpf_delta_std'],
                                stage_stats['ahpf_strength_mean'],
                                stage_stats['ahpf_identity_weight'],
                                stage_stats['ahpf_blur3_weight'],
                                stage_stats['ahpf_blur5_weight'],
                                stage_stats['ahpf_kernel_sum'],
                                stage_stats['ahpf_kernel_entropy'],
                                stage_stats['ahpf_residual_std'],
                                stage_stats['adaptive_skip_residual_std'],
                            )
                        )
                    continue
                logger.info(
                    '{} guide_mix: entropy={:.4f}, skip={:.4f}, '
                    'decoder={:.4f}, local_edge={:.4f}, '
                    'context_edge={:.4f}'.format(
                        stage,
                        stage_stats['guide_branch_entropy'],
                        stage_stats['guide_skip_weight'],
                        stage_stats['guide_decoder_weight'],
                        stage_stats['guide_local_edge_weight'],
                        stage_stats['guide_context_edge_weight'],
                    )
                )
                if 'low_pass_kernel_sum' in stage_stats:
                    logger.info(
                        '{} plam_frequency: pixel_std={:.4f}, '
                        'edge_std={:.4f}, support={:.4f}, '
                        'kernel_sum={:.4f}, kernel_entropy={:.4f}, '
                        'kernel_center={:.4f}, kernel_delta={:.4e}'.format(
                            stage,
                            stage_stats['pixel_residual_std'],
                            stage_stats['edge_residual_std'],
                            stage_stats['semantic_support_mean'],
                            stage_stats['low_pass_kernel_sum'],
                            stage_stats['low_pass_kernel_entropy'],
                            stage_stats['low_pass_kernel_center'],
                            stage_stats['low_pass_kernel_delta_abs'],
                        )
                    )

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    print("[boot] entered __main__, sending Bark start notification...", flush=True)
    bark_notify("模型开始训练了，请耐心等待！", title="🚀 训练开始")
    print("[boot] Bark call returned, continuing setup...", flush=True)
    configure_determinism(config.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            'Strict paired training requires exactly one visible CUDA GPU'
        )
    if config.persistent_workers:
        raise RuntimeError(
            'Strict epoch-boundary resume requires persistent_workers=False'
        )
    if len(config.git_commit) != 40:
        raise RuntimeError(
            'BETTERLVIT_GIT_COMMIT must identify the reviewed source commit'
        )
    verify_source_checkout(config.git_commit)
    output_directory = Path(config.save_path).resolve()
    if output_directory.exists():
        raise FileExistsError(
            'Refusing to reuse session directory: {}'.format(
                output_directory
            )
        )
    output_directory.mkdir(parents=True, exist_ok=False)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
    bark_notify("训练完成！", title="✅ 训练结束")
    if getattr(config, 'shutdown_after_training', False):
        print("Training complete. Shutting down in 60 seconds...")
        os.system("shutdown /s /t 60")
