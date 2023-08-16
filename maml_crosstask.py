import os
import logging

import numpy as np
import wandb
import torch
from torch import nn, optim
import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)

from backbones.conv4 import Conv4
from utils.seed import seed_everything
from utils.load_data import get_tasksets
from utils.adapt import fast_adapt_maml
from utils.mix import crosstask_mixup
from configs.maml_crosstask_miniImageNet_S import Config


def main(config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id
    logger.info(f'DEVICE: {config.device}, GPU ID: {config.gpu_id}')

    os.makedirs(config.ckpt_dir, exist_ok=True)

    seed_everything(config.seed)
    logger.info(f'SEED: {config.seed}')
    
    if config.use_wandb:
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=f"{config.name}",
            config={k: v for k, v in config.__dict__.items() if not callable(v) and not k.startswith("__")}
        )

    logger.info(f'LOADING TASKSETS...')
    tasksets = get_tasksets(config)
    logger.info(f'TASKSETS LOADED.')

    model = Conv4(config.x_dim, config.hid_dim, config.z_dim, config.ways)
    model.to(config.device)
    maml = l2l.algorithms.MAML(model, lr=config.fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), config.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    best_test_accuracy = 0.0

    logger.info(f'TRAINING PHASE HAS STARTED')
    logger.info(f'MIXUP METHOD: {config.mixup_method}')

    for iteration in range(config.num_iterations):
        opt.zero_grad()
        meta_train_error, meta_train_accuracy = 0.0, 0.0
        meta_valid_error, meta_valid_accuracy = 0.0, 0.0
        meta_test_error, meta_test_accuracy = 0.0, 0.0
        for task in range(config.meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch1 = tasksets.train.sample()
            batch2 = tasksets.train.sample()
            batch = crosstask_mixup(batch1, batch2, config.device, config.mixup_method)

            evaluation_error, evaluation_accuracy = \
                fast_adapt_maml(batch,
                                learner,
                                loss,
                                config.adaptation_steps,
                                config.shots,
                                config.ways,
                                config.device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = \
                fast_adapt_maml(batch,
                                learner,
                                loss,
                                config.adaptation_steps*2, # and evaluated using 10 gradient steps at test time. 
                                config.shots,
                                config.ways,
                                config.device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-test loss
            learner = maml.clone()
            batch = tasksets.test.sample()
            evaluation_error, evaluation_accuracy = \
                fast_adapt_maml(batch,
                                learner,
                                loss,
                                config.adaptation_steps*2, # and evaluated using 10 gradient steps at test time. 
                                config.shots,
                                config.ways,
                                config.device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()            

        # Print some metrics
        meta_train_error /= config.meta_batch_size
        meta_train_accuracy /= config.meta_batch_size
        meta_valid_error /= config.meta_batch_size
        meta_valid_accuracy /= config.meta_batch_size
        meta_test_error /= config.meta_batch_size
        meta_test_accuracy /= config.meta_batch_size

        print("\x1B[H\x1B[J")
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error)
        print('Meta Train Accuracy', meta_train_accuracy)
        print('Meta Valid Error', meta_valid_error)
        print('Meta Valid Accuracy', meta_valid_accuracy)
        print('Meta Test Error', meta_test_error)
        print('Meta Test Accuracy', meta_test_accuracy)

        if config.use_wandb:
            wandb.log({
                "Meta Train Error": meta_train_error, 
                "Meta Train Accuracy": meta_train_accuracy,
                "Meta Valid Error": meta_valid_error,
                "Meta Valid Accuracy": meta_valid_accuracy,
                "Meta Test Error": meta_test_error,
                "Meta Test Accuracy": meta_test_accuracy,
                })
        
        ckpt_metric = meta_test_accuracy
        if ckpt_metric > best_test_accuracy:
            best_test_accuracy = ckpt_metric
            torch.save(maml.module.state_dict(), f'{config.ckpt_dir}/best.pt')
            print('Checkpoint Saved.')

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / config.meta_batch_size)
        opt.step()
    
    print('Best Test Accuracy', best_test_accuracy)

    if config.use_wandb:
        wandb.log({
            "Best Test Accuracy": best_test_accuracy,
        })


if __name__ == '__main__':
    config = Config
    main(config=config)
