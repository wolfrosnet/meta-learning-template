import datetime as dt

import torch


class Config:
    algorithm='mlti'
    dataset_name='miniImageNet-S'
    ways=5
    shots=5
    meta_lr=0.001
    fast_lr=0.01 # For MiniImagenet, both models were trained using 5 gradient steps of size α = 0.01, 
    meta_batch_size=4 # We used a meta batch-size of 4 and 2 tasks for 1-shot and 5-shot training respectively.
    adaptation_steps=5 # For MiniImagenet, both models were trained using 5 gradient steps of size α = 0.01, 
    num_iterations=60000 # All models were trained for 60000 iterations on a single NVIDIA Pascal Titan X GPU.
    seed=42
    time_now = dt.datetime.now()
    run_id = time_now.strftime("%m%d%H%M%S")
    device = torch.device('cuda')
    gpu_id='7'
    name = f'{algorithm}_{dataset_name}_{run_id}'
    ckpt_dir = f'./checkpoint/{name}'
    wandb_entity = 'auroraveil' # WandB nickname
    wandb_project = 'mamlmamlmaml' # WandB project name
    x_dim=3
    hid_dim=32 # For MiniImagenet, we used 32 filters per layer to reduce overfitting, as done by (Ravi & Larochelle, 2017). 
    z_dim=32
    use_wandb=True
    mixup_method='cutmix'
    test_ckpt_path='./checkpoint/mlti_miniImageNet-S_0821151020/best.pt'
    num_test_points=600
