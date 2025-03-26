from . import utils, layers, normalization

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Configuration


get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

NONLINEARITIES = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "lrelu": nn.LeakyReLU(negative_slope=0.2),
    "swish": nn.SiLU(),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
}


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(IgnoreLinear).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self._layer(x))

class BlendLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(BlendLinear, self).__init__()
        self._layer0 = nn.Linear(dim_in, dim_out)
        self._layer1 = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        out = y0 + (y1 - y0) * t[:, None]
        out = self.bn(out)
        return out
    
class ConcatLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tt = torch.ones_like(x[:, :1]) * t[:, None]
        ttx = torch.cat([tt, x], 1)

        return self._layer(ttx)

class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))

class SquashLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(-1, 1)))
    
class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) + self._hyper_bias(t.view(-1, 1))
    
class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def get_sigmas(config: Configuration):
    """ Get sigmas --- the set of noise levels for SMLD from config.

    Args:
        config: A ConfigDict object parsed from the config file
    
    Returns:
        sigmas: a jax numpy arrary of noise levels
    """
    return np.exp(
    np.linspace(np.log(config.model_sigma_max), np.log(config.model_sigma_min), config.model_num_scales))

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: 
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

@utils.register_model(name='ncsnpp_tabular')
class NCSNpp(nn.Module):
    """NCSN++ model."""

    def __init__(self, config: Configuration):
        super().__init__()

        base_layer =  {
            "ignore": IgnoreLinear,
            "squash": SquashLinear,
            "concat": ConcatLinear,
            "concat_v2": ConcatLinear_v2,
            "concatsquash": ConcatSquashLinear,
            "blend": BlendLinear,
            "concatcoord": ConcatLinear,
        }

        self.config = config

        self.act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(get_sigmas(config)))
        self.hidden_dims = config.model_hidden_dims 
        self.nf = nf = config.model_nf

        self.conditional = conditional = config.model_conditional 
        self.embedding_type = embedding_type = config.model_embedding_type.lower()

        modules = []
        if embedding_type == 'fourier':
            assert config.training_continuous, "Fourier features are only used for continuous training."

            modules.append(GaussianFourierProjection(embedding_size=nf, scale=config.model_fourier_scale))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
        
        dim = config.data_image_size
        for item in list(config.model_hidden_dims):
            modules += [base_layer[config.model_layer_type](dim, item)]
            dim += item
            modules.append(NONLINEARITIES[config.model_activation])

        modules.append(nn.Linear(dim, config.data_image_size))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor):
        modules = self.all_modules 
        m_idx = 0
        if self.embedding_type == 'fourier':
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            used_sigmas = self.sigmas[time_cond.long()]
            temb = get_timestep_embedding(time_cond, self.nf)

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        temb = x
        for _ in range(len(self.hidden_dims)):
            temb1 = modules[m_idx](t=time_cond, x=temb)
            temb = torch.cat([temb1, temb], dim=1)
            m_idx += 1
            temb = modules[m_idx](temb) 
            m_idx += 1

        h = modules[m_idx](temb)

        if self.config.model_scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h

class MLPBlock(nn.Module):
    def __init__(self, d_in, d_out, bias=True, dropout=0.0):
        super().__init__()

        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))

class GLUBlock(nn.Module):
    def __init__(self, d_in, d_out, bias=True, dropout=0.0):
        super().__init__()
        self.linear_A = nn.Linear(d_in, d_out, bias=bias)
        self.linear_B = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear_A(x) * self.activation(self.linear_B(x)))

class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_layers: list[int], dropout=0.0):
        super().__init__()
        
        self.blocks = nn.ModuleList(
            [
                # MLPBlock(
                GLUBlock(
                    d_in=d_layers[i - 1] if i else d_in, 
                    d_out=d, 
                    dropout=dropout
                ) 
                for i, d in enumerate(d_layers)
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x

def timestep_embedding(timesteps: torch.Tensor, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    Return: 
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

@utils.register_model(name='ddpm_tabular')
class DDPM(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        self.dim_t = 128
        d_in = self.dim_t
        d_layers = config.model_hidden_dims
        d_out = config.data_image_size
        self.mlp = MLP(d_in, d_out, d_layers)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.dim_t, self.dim_t),
            nn.SiLU(),
            nn.Linear(self.dim_t, self.dim_t)
        )
        self.proj = nn.Linear(config.data_image_size, self.dim_t)
    
    def forward(self, x: torch.Tensor, time_cond: torch.Tensor):
        emb = self.time_embedding(timestep_embedding(time_cond, self.dim_t))
        x = self.proj(x) + emb
        return self.mlp(x)

