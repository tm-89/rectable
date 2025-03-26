"""Training Rectified Flow"""

import argparse
from typing import Optional, Literal
from pydantic import BaseModel, Field

class Configuration(BaseModel):

    # training
    training_batch_size: Optional[int] = Field(4096)
    training_n_iters: Optional[int] = Field(30000)
    training_likelihood_weighting: Literal[True, False] = Field(False)
    training_continuous: Literal[True, False] = Field(True)
    training_reduce_mean: Literal[True, False] = Field(True)
    training_sde: Optional[str] = Field('rectified_flow')
    training_snapshot_freq: Optional[int] = Field(1000)

    # sampling
    sampling_n_steps_each: Optional[int] = Field(1)
    sampling_noise_removal: Literal[True, False] = Field(True)
    sampling_probability_flow: Literal[True, False] = Field(False)
    sampling_snr: Optional[float] = Field(0.075)
    sampling_sigma_variance: Optional[float] = Field(0.0) # NOTE: XC: sigma variance for turning ODe to SDE
    sampling_init_noise_scale: Optional[float] = Field(1.0)
    sampling_use_ode_sampler: Optional[str] = Field('rk45')
    sampling_ode_tol: Optional[float] = Field(1e-5)
    sampling_sample_N: Optional[int] = Field(1000)
    sampling_method: Optional[str] = Field('rectified_flow')
    sampling_init_type: Literal["gaussian", "concat"] = Field('concat')


    # evaluation
    evaluate_begin_ckpt: Optional[int] = Field(50)
    evaluate_end_ckpt: Optional[int] = Field(96)
    evaluate_batch_size: Optional[int] = Field(512)
    evaluate_enable_sampling: Literal[True, False] = Field(False)
    evaluate_enable_figures_only: Literal[True, False] = Field(False)
    evaluate_num_samples: Optional[int] = Field(50000)
    evaluate_enable_loss: Literal[True, False] = Field(False)
    evaluate_enable_bpd: Literal[True, False] = Field(False)
    evaluate_bpd_dataset: Optional[str] = Field('test')

    # data
    data_dataset: Optional[str] = Field('adult')
    data_image_size: Optional[int] = Field(77)
    data_use_resbit: Literal[True, False] = Field(False)
    data_scale_ohe: Literal[True, False] = Field(True)

    # model
    model_sigma_max: Optional[float] = Field(10)
    model_sigma_min: Optional[float] = Field(0.01)
    model_num_scales: Optional[int] = Field(50)
    model_beta_min: Optional[float] = Field(0.01)
    model_beta_max: Optional[float] = Field(20.)
    model_dropout: Optional[float] = Field(0.)
    model_embedding_type: Optional[str] = Field('fourier')

    model_alpha0: Optional[float] = Field(0.3)
    model_beta0: Optional[float] = Field(0.95)
    model_layer_type: Optional[str] = Field('concatsquash')
    model_name: Literal["ddpm_tabular", "ncsnpp_tabular"] = Field('ddpm_tabular')
    model_scale_by_sigma: Literal[True, False] = Field(False)
    model_ema_rate: Optional[float] = Field(0.9999)
    model_activation: Optional[str] = Field('elu')

    model_nf: Optional[int] = Field(64)
    model_hidden_dims: Optional[list[int]] = Field([1024, 2048, 1024, 1024])
    model_conditional: Literal[True, False] = Field(True)
    model_fourier_scale: Optional[int] = Field(16)
    model_conv_size: Optional[int] = Field(3)

    # optimization
    optim_weight_decay: Optional[float] = Field(0)
    optim_optimizer: Optional[str] = Field('Adam')
    optim_lr: Optional[float] = Field(2e-4)
    optim_beta1: Optional[float] = Field(0.9)
    optim_eps: Optional[float] = Field(1e-8)
    optim_warmup: Optional[int] = Field(5000)
    optim_grad_clip: Optional[float] = Field(1.)

    # reflow
    reflow_reflow_type: Optional[str] = Field('train_reflow')
    reflow_reflow_t_schedule: Optional[str] = Field('uniform')
    reflow_reflow_loss: Optional[str] = Field('l2')
    reflow_last_flow_ckpt: Optional[str] = Field('checkpoints/checkpoint_029999.pth')
    reflow_data_root: Optional[str] = Field('data_path')
    reflow_u_shape_td_a: Optional[int] = Field(4)

    seed: Optional[int] = Field(42)
    device: Optional[str] = Field("cpu")
    
    dataname: Optional[str] = Field("adult")
    exp: Optional[str] = Field("")

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        fields = cls.model_fields
        for name, field in fields.items():
            parser.add_argument(f"--{name}", default=field.default, help=field.description)
        return cls.model_validate(parser.parse_args().__dict__)
