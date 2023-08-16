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
from configs.maml_miniImageNet_S import Config


def main(config):
    ckpt_path = './checkpoint/somewhere/best.pt'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id
    logger.info(f'DEVICE: {config.device}, GPU ID: {config.gpu_id}')

    seed_everything(config.seed)
    logger.info(f'SEED: {config.seed}')

    logger.info(f'LOADING TASKSETS...')
    tasksets = get_tasksets(config)
    logger.info(f'TASKSETS LOADED.')

    model = Conv4(config.x_dim, config.hid_dim, config.z_dim, config.ways)
    model.to(config.device)
    maml = l2l.algorithms.MAML(model, lr=config.fast_lr, first_order=False)
    loss = nn.CrossEntropyLoss(reduction='mean')

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    model.to(config.device)
    logger.info(f'Checkpoint Loaded: {ckpt_path}')


    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(config.meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt_maml(batch,
                                                            learner,
                                                            loss,
                                                            config.adaptation_steps*2,
                                                            config.shots,
                                                            config.ways,
                                                            config.device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()

    meta_test_error /= config.meta_batch_size
    meta_test_accuracy /= config.meta_batch_size

    print('Meta Test Error', meta_test_error)
    print('Meta Test Accuracy', meta_test_accuracy)


if __name__ == '__main__':
    config = Config
    main(config=config)
