"""Full-mask extent supervision and separate regional presence supervision."""
import torch
from torch import nn
from torch.nn import functional as F
from utils import WeightedDiceFocal


def masked_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1)


class RACEPEObjective(nn.Module):
    def __init__(self, aux_weight=0.05, pixel_only=False, drop_report_consistency=False, **kwargs):
        super().__init__()
        self.segmentation = WeightedDiceFocal(**kwargs)
        self.aux_weight = aux_weight
        self.pixel_only = pixel_only
        self.drop_report_consistency = drop_report_consistency
        self.last_components = {}

    def _show_dice(self, inputs, targets):
        return self.segmentation._show_dice(inputs, targets)

    def forward(self, outputs, targets):
        main = self.segmentation(outputs["final"], targets)
        slots, labels = outputs["slot_logits"], outputs["race_slot_targets"]
        mask = targets.float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("Expected binary masks [B,1,H,W] or [B,H,W]")
        zone_known = labels[:, :6] >= 0
        text_loss = masked_mean(F.binary_cross_entropy_with_logits(
            slots[:, :6], labels[:, :6].clamp_min(0), reduction="none"
        ), zone_known)
        count_known = (labels[:, 6:] >= 0).all(1)
        count_loss = masked_mean(F.cross_entropy(
            slots[:, 6:], labels[:, 6:].argmax(1), reduction="none"
        ), count_known)
        pixel_losses, presence_losses, occupancy_losses, consistency_losses = [], [], [], []
        for route in outputs["pe_routes"]:
            logits, basis = route["extent_logits"], route["basis"]
            # Area-resampled soft targets preserve small-lesion occupancy.
            target = F.adaptive_avg_pool2d(mask, logits.shape[-2:])
            pixel_losses.append(F.binary_cross_entropy_with_logits(logits, target))
            mass = basis.sum((2, 3))
            valid = mass > 0
            occupancy = (target * basis).sum((2, 3)) / mass.clamp_min(1)
            # Regional existence is supervised on its own head, never mean extent.
            present = (occupancy > 0).float()
            presence_losses.append(masked_mean(F.binary_cross_entropy_with_logits(
                route["presence_logits"], present, reduction="none"
            ), valid))
            occupancy_losses.append(masked_mean(
                (route["occupancy"] - occupancy).square(), valid
            ))
            positive = (labels[:, :6] == 1) & valid
            consistency_losses.append(masked_mean(
                F.softplus(-route["presence_logits"]), positive
            ))
        pixel = torch.stack(pixel_losses).mean()
        presence = torch.stack(presence_losses).mean()
        occupancy_loss = torch.stack(occupancy_losses).mean()
        consistency = torch.stack(consistency_losses).mean()
        auxiliary = 0.4 * pixel if self.pixel_only else (
            0.4 * pixel + 0.2 * presence + 0.1 * occupancy_loss
            + 0.2 * (0.75 * text_loss + 0.25 * count_loss)
            + (0.0 if self.drop_report_consistency else 0.1) * consistency
        )
        self.last_components = {"main": main.detach(), "pe_pixel": pixel.detach(),
            "pe_presence": presence.detach(), "pe_occupancy": occupancy_loss.detach(),
            "pe_text": text_loss.detach(), "pe_consistency": consistency.detach(),
            "pe_aux_weighted": (self.aux_weight * auxiliary).detach()}
        return main + self.aux_weight * auxiliary
