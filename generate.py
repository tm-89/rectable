import os
import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from models import ncsnpp
import losses
import sampling
import sde_lib
import src
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from config import Configuration
from utils_train import preprocess


def restore_checkpoint(ckpt_dir, state, device):
    if not Path(ckpt_dir).exists():
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state

def recover_data(syn_num: np.ndarray, syn_cat: np.ndarray, info: dict[str, Any]):

    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df


def generate(args):

    config = Configuration()

    dataname = args.dataname
    config.data_dataset = dataname
    dataset_dir = f'data/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    num_numericals = len(info["num_col_idx"])
    dataset, K = preprocess(dataset_dir, task_type=task_type, cat_encoding='one-hot', use_resbit=config.data_use_resbit)
    train_z = torch.tensor(dataset.X_num['train'])
    config.data_image_size = train_z.shape[1]

    # Initialize model
    score_model: torch.nn.Module = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model_ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = "checkpoints"

    # Setup SDEs
    sde = sde_lib.RectifiedFlow(
        init_type=config.sampling_init_type, 
        noise_scale=config.sampling_init_noise_scale, 
        use_ode_sampler=config.sampling_use_ode_sampler,
        sigma_var=config.sampling_sigma_variance,
        ode_tol=config.sampling_ode_tol,
        sample_N=train_z.shape[0],
        num_numericals=num_numericals,
        scale_ohe=config.data_scale_ohe
    )
    sampling_eps = 1e-3


    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{args.ckpt}.pth")
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    config.evaluate_batch_size = train_z.shape[0]
    sampling_shape = (config.evaluate_batch_size, config.data_image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler=None, eps=sampling_eps)

    samples, nfe = sampling_fn(score_model)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_data_num = samples[:, :n_num_feat].cpu()
    cat_sample = samples[:, n_num_feat:].cpu()

    # resbit -> one-hot
    if config.data_use_resbit:
        l = src.get_length_resbit(K)
        cat_sample = src.resbit_to_ohe(cat_sample, l, K)

    if config.data_scale_ohe:
        if isinstance(cat_sample, torch.Tensor):
            cat_sample = cat_sample.numpy()
        cat_sample = (cat_sample > 0).astype(float)

    num_inverse = dataset.num_transform.inverse_transform
    cat_inverse = dataset.cat_transform.inverse_transform

    syn_num = num_inverse(syn_data_num)
    syn_cat = cat_inverse(cat_sample)

    syn_df = recover_data(syn_num, syn_cat, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    os.makedirs(f"synthetic/{args.dataname}", exist_ok=True)
    if args.trial_num is None:
        args.trial_num = "0"
    syn_df.to_csv(f"synthetic/{args.dataname}/{args.ckpt}/{args.trial_num}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='001000')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--trial_num', type=str, default=None)
    args = parser.parse_args()

    generate(args)

