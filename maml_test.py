import os
import logging

from tqdm.auto import tqdm
import torch
from torch import nn
import learn2learn as l2l

from backbones.conv4 import Conv4
from utils.seed import seed_everything
from utils.load_data import get_tasksets
from utils.adapt import fast_adapt_maml
from configs.maml_miniImageNet_S import Config


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

    seed_everything(config.seed)
    logger.info(f'SEED: {config.seed}')

    logger.info(f'LOADING TASKSETS...')
    tasksets = get_tasksets(config)
    logger.info(f'TASKSETS LOADED.')

    model = Conv4(config.x_dim, config.hid_dim, config.z_dim, config.ways)
    model.to(config.device)
    maml = l2l.algorithms.MAML(model, lr=config.fast_lr, first_order=False)
    loss = nn.CrossEntropyLoss(reduction='mean')

    state_dict = torch.load(config.test_ckpt_path)
    model.load_state_dict(state_dict)
    model.to(config.device)
    logger.info(f'Checkpoint Loaded: {config.test_ckpt_path}')

    with open("./etc/phase.txt", "r") as f:
        lines = f.readlines()
    phase_text = ''.join(lines[11:19])
    print(phase_text)

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in tqdm(range(config.num_test_points)):
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

    meta_test_error /= config.num_test_points
    meta_test_accuracy /= config.num_test_points

    print('Meta Test Error', meta_test_error)
    print('Meta Test Accuracy', meta_test_accuracy)


if __name__ == '__main__':
    config = Config
    main(config=config)
