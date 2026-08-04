"""Run two real AMD training steps for the boundary-loss experiment."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torchvision import transforms

import Config as config
from Load_Dataset import ImageToImage2D, RandomGenerator
from nets.BetterLViT import BetterLViT
from utils import WeightedDiceBCE, read_text


def build_model():
    return BetterLViT(
        config.get_CTranS_config(),
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


def main():
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("An AMD ROCm PyTorch environment is required.")

    text = read_text(
        os.path.join(config.task_dataset, "Train_Val_text.xlsx")
    )
    dataset = ImageToImage2D(
        config.train_dataset,
        config.task_name,
        text,
        transforms.Compose(
            [RandomGenerator([config.img_size, config.img_size])]
        ),
        image_size=config.img_size,
    )
    samples = [
        dataset[index][0] for index in range(config.batch_size)
    ]
    images = torch.stack(
        [sample["image"] for sample in samples]
    ).cuda()
    labels = torch.stack(
        [sample["label"] for sample in samples]
    ).float().cuda()
    input_ids = torch.stack(
        [sample["input_ids"] for sample in samples]
    ).cuda()
    attention_mask = torch.stack(
        [sample["attention_mask"] for sample in samples]
    ).cuda()

    model = build_model().cuda().train()
    criterion = WeightedDiceBCE(
        dice_weight=0.5,
        BCE_weight=0.5,
        boundary_weight=config.boundary_loss_weight,
        boundary_kernel_size=config.boundary_kernel_size,
    )
    optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    torch.cuda.reset_peak_memory_stats()
    for step in range(1, 3):
        predictions = model(images, input_ids, attention_mask)
        loss = criterion(predictions, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        output_gradient = model.outc.weight.grad
        edge_gradient = model.up1.eppa.sp_proj.weight.grad
        for name, gradient in (
            ("output head", output_gradient),
            ("EPPA edge branch", edge_gradient),
        ):
            if (
                gradient is None
                or not torch.isfinite(gradient).all()
                or gradient.abs().max().item() == 0
            ):
                raise RuntimeError(f"{name} gradient is invalid")
        if not torch.isfinite(loss):
            raise RuntimeError("boundary objective produced a non-finite loss")
        optimizer.step()
        print(
            f"step={step} loss={loss.item():.6f} "
            f"components={criterion.last_components}",
            flush=True,
        )

    torch.cuda.synchronize()
    print("GPU:", torch.cuda.get_device_name(0))
    print("Batch:", config.batch_size)
    print("Prediction:", tuple(predictions.shape))
    print(
        "Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB"
    )
    print("Boundary-aware training check passed.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        os._exit(0)
