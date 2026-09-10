"""Repair prototype: derive one region target from matching full-resolution geometry."""
import torch


@torch.no_grad()
def full_resolution_region_targets(mask,transformed_zone_basis):
    if mask.ndim==3:mask=mask.unsqueeze(1)
    if mask.ndim!=4 or mask.shape[1]!=1 or transformed_zone_basis.ndim!=4:
        raise ValueError('Expected B1HW mask and BZHW transformed region basis')
    if mask.shape[0]!=transformed_zone_basis.shape[0] or mask.shape[-2:]!=transformed_zone_basis.shape[-2:]:
        raise ValueError('Targets must use matching full-resolution mask and transformed basis')
    mask=mask.float();basis=transformed_zone_basis.float()
    mass=basis.sum((2,3));foreground_mass=(mask*basis).sum((2,3))
    valid=mass>0
    return dict(area=foreground_mass/mass.clamp_min(1),present=(foreground_mass>0).float(),valid=valid)
