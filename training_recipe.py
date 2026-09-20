"""Registered training recipes; no model architecture or loss changes."""
import math


def planned_rates(schedule, epochs=80, maximum=3e-4):
    if schedule == 'warm_restarts':
        return [1e-4 + .5*(maximum-1e-4)*(1+math.cos(math.pi*(e%10)/10)) for e in range(epochs)]
    if schedule == 'single_cosine':
        return [maximum if e == 0 else 1e-6 if e == epochs-1 else
                1e-6 + .5*(maximum-1e-6)*(1+math.cos(math.pi*e/(epochs-1))) for e in range(epochs)]
    raise ValueError('Unknown schedule')


def rates_equal(actual, expected):
    """Allow libm rounding across Windows/Linux, not a changed LR recipe."""
    return len(actual) == len(expected) and all(
        math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-14, abs_tol=0.0)
        for a, b in zip(actual, expected))


def validate_recipe_manifest(manifest):
    policies = {'r1_chest_augmentation':('chest_orientation','warm_restarts'),
                'r2_single_cosine':('legacy','single_cosine')}
    expected = policies[manifest['profile']]
    assert (manifest['augmentation_policy'],manifest['lr_schedule']) == expected
    assert manifest['epochs'] == 80 and manifest['batch_size'] == 16
    assert manifest['selection_metric'] == 'iou' and manifest['threshold'] == .5
    assert manifest['loss_name'] == 'dice_focal' and manifest['initialization'] == 'from_scratch'
    assert not any(manifest[k] for k in ('test_split_allowed','auto_test_evaluate','lora','boundary_loss','new_supervision','visual_prior'))
    assert rates_equal(manifest['planned_epoch_lrs'], planned_rates(expected[1]))


def recipe_metadata(config):
    if not config.training_recipe_enabled:
        return None
    return {'version':'c4_recipe_v1', 'augmentation_policy':config.augmentation_policy,
        'lr_schedule':config.lr_schedule, 'epochs':config.epochs,
        'lr_max':config.learning_rate,
        'lr_min':1e-6 if config.lr_schedule == 'single_cosine' else 1e-4,
        'optimizer':'Adam', 'weight_decay':config.weight_decay,
        'planned_epoch_lrs':planned_rates(config.lr_schedule,config.epochs,config.learning_rate),
        'lr_epoch_convention':'first and last training epochs include endpoints' if config.lr_schedule == 'single_cosine' else 'legacy T_0=10 T_mult=1'}


class SingleCosineSchedule:
    """After validation at epoch e, set the LR for training epoch e+1.

    Index zero is the first epoch. The last training epoch reaches eta_min;
    further steps clamp there. State restoration also restores optimizer LRs.
    """
    def __init__(self, optimizer, epochs, eta_min=1e-6):
        if epochs < 2:
            raise ValueError('At least two epochs are required')
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.eta_min = float(eta_min)
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        if not all(base >= self.eta_min >= 0 for base in self.base_lrs):
            raise ValueError('Invalid learning rate endpoints')
        self.last_epoch = 0
        self._apply()

    def _apply(self):
        index = min(self.last_epoch, self.epochs-1)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            if index == 0:
                rate = base
            elif index == self.epochs-1:
                rate = self.eta_min
            else:
                rate = self.eta_min + .5*(base-self.eta_min)*(1+math.cos(math.pi*index/(self.epochs-1)))
            group['lr'] = rate

    def step(self, epoch=None):
        self.last_epoch = self.last_epoch+1 if epoch is None else int(epoch)
        self._apply()

    def state_dict(self):
        return {key:getattr(self,key) for key in ('epochs','eta_min','base_lrs','last_epoch')}

    def load_state_dict(self, state):
        if state['epochs'] != self.epochs or state['eta_min'] != self.eta_min or state['base_lrs'] != self.base_lrs:
            raise ValueError('Schedule configuration mismatch')
        self.last_epoch = int(state['last_epoch'])
        self._apply()
