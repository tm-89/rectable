import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import utils as mutils
from sde_lib import RectifiedFlow


def get_optimizer(config, params: nn.Parameter):
    """Returns a flax optimizer object based on `config`."""
    if config.optim_optimizer == 'Adam':
        optimizer = optim.Adam(
           params, 
           lr=config.optim_lr, 
           betas=(config.optim_beta1, 0.999), 
           eps=config.optim_eps,
           weight_decay=config.optim_weight_decay
        )
    elif config.optim_optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params,
            lr=config.optim_lr,
            betas=(config.optim_beta1, 0.999),
            eps=config.optim_eps,
            weight_decay=config.optim_weight_decay
        )
    else:
        raise NotImplementedError(f'Optimizer {config.optim_optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer: optim.Optimizer, params, step, lr=config.optim_lr, warmup=config.optim_warmup, grad_clip=config.optim_grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0 and config.optim_optimizer != 'RAdamScheduleFree':
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)
    
def sample_t(exponential_distribution: ExponentialPDF, num_samples, a):
    t = exponential_distribution.rvs(size=num_samples, a=a)
    t = torch.from_numpy(t).float()
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def pseudo_hurber(x: torch.Tensor, y: torch.Tensor):
    data_dim = x.shape[1]
    huber_c = 0.00054 * data_dim
    loss = torch.sum((x - y) ** 2, dim=-1)
    loss = torch.sqrt(loss + huber_c ** 2) - huber_c
    return loss / data_dim

def get_rectified_flow_loss_fn(sde: RectifiedFlow, train, reduce_mean=True, eps=1e-3):
    """Create a loss function for training with rectified flow.

    Args:
        sde: An `sde_lib.RectifiedFlow` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch: torch.Tensor):
        """Compute the loss function.

        Args:
            model: A velocity model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        
        z0 = sde.get_z0(batch).to(batch.device)
    
        if sde.reflow_flag:
            if sde.reflow_t_schedule == 't0': ### distill for t = 0 (k=1)
                t = torch.zeros(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            elif sde.reflow_t_schedule == 't1': ### reverse distill for t=1 (fast embedding)
                t = torch.ones(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            elif sde.reflow_t_schedule == 'uniform': ### train new rectified flow with reflow
                t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            elif type(sde.reflow_t_schedule) == int: ### k > 1 distillation
                t = torch.randint(0, sde.reflow_t_schedule, (batch.shape[0], ), device=batch.device) * (sde.T - eps) / sde.reflow_t_schedule + eps
            elif sde.reflow_t_schedule == "u_shape":
                exponential_distribution = ExponentialPDF(a=0, b=1)
                t = sample_t(exponential_distribution, batch.shape[0], sde.u_shape_td_a).to(batch.device)
            elif sde.reflow_t_schedule == "lognorm":
                u = torch.normal(mean=0, std=1.0, size=(batch.shape[0],), device=batch.device)
                u = torch.nn.functional.sigmoid(u)
                t = u * (sde.T - eps) + eps
            else:
                assert False, 'Not implemented'
        else:
            ### standard rectified flow loss
            # t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps

            # lognorm(0,1)
            t = torch.nn.functional.sigmoid(torch.normal(mean=0, std=1, size=(batch.shape[0],), device=batch.device)) * (sde.T - eps) + eps

        t_expand = t.view(-1, 1).repeat(1, batch.shape[1])
        perturbed_data = t_expand * batch + (1.-t_expand) * z0
        target = batch - z0 
    
        model_fn = mutils.get_model_fn(model, train=train)
        score = model_fn(perturbed_data, t*999) ### Copy from models/utils.py 

        if sde.reflow_flag:
            ### we found LPIPS loss is the best for distillation when k=1; but good to have a try
            if sde.reflow_loss == 'l2':
                ### train new rectified flow with reflow or distillation with L2 loss
                losses = torch.square(score - target)
            elif sde.reflow_loss == "hurber":
                losses = pseudo_hurber(score, target)
            else:
                assert False, 'Not implemented'
        else:
            if not sde.switch_loss_to_hurber:
                losses = torch.square(score - target)
            else:
                losses = pseudo_hurber(score, target)
        
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(sde: RectifiedFlow, train, optimize_fn=None, reduce_mean=False, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.RectifiedFlow` object that represents the forward SDE.
        optimize_fn: An optimization function.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

      Returns:
        A one-step function for training or evaluation.
    """
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, RectifiedFlow):
        loss_fn = get_rectified_flow_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
            state: A dictionary of training information, containing the score model, optimizer,
            EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data.

        Returns:
            loss: The average loss value of this state.
        """
        model: nn.Module = state['model']
        if train:
            optimizer: optim.Optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn