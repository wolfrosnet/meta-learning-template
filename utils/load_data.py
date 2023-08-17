from collections import namedtuple

from torchvision import datasets, transforms
import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)


def get_tasksets(config):
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.ImageFolder(f'/data/mghong/dataset/{config.dataset_name}/train', transform)
    validset = datasets.ImageFolder(f'/data/mghong/dataset/{config.dataset_name}/val', transform)
    testset = datasets.ImageFolder(f'/data/mghong/dataset/{config.dataset_name}/test', transform)

    metadataset_train = l2l.data.MetaDataset(trainset)
    metadataset_valid = l2l.data.MetaDataset(validset)
    metadataset_test = l2l.data.MetaDataset(testset)

    train_tasks = l2l.data.Taskset(
        metadataset_train,
        task_transforms=[
            NWays(metadataset_train, config.ways),
            KShots(metadataset_train, 2*config.shots),
            LoadData(metadataset_train),
            RemapLabels(metadataset_train),
            ConsecutiveLabels(metadataset_train),
        ],
        num_tasks=-1,
    )
    valid_tasks = l2l.data.Taskset(
        metadataset_valid,
        task_transforms=[
            NWays(metadataset_valid, config.ways),
            KShots(metadataset_valid, 2*config.shots),
            # KShots(metadataset_valid, 15*2), # 15 examples per class were used for evaluating the post-update meta-gradient.
            LoadData(metadataset_valid),
            RemapLabels(metadataset_valid),
            ConsecutiveLabels(metadataset_valid),
        ],
        num_tasks=-1,
    )
    test_tasks = l2l.data.Taskset(
        metadataset_test,
        task_transforms=[
            NWays(metadataset_test, config.ways),
            l2l.data.transforms.KShots(metadataset_test, 2*config.shots),
            # KShots(metadataset_test, 15*2), # 15 examples per class were used for evaluating the post-update meta-gradient.
            LoadData(metadataset_test),
            RemapLabels(metadataset_test),
            ConsecutiveLabels(metadataset_test),
        ],
        num_tasks=-1,
    )

    BenchmarkTasksets = namedtuple('BenchmarkTasksets', ('train', 'validation', 'test'))
    tasksets = BenchmarkTasksets(train_tasks, valid_tasks, test_tasks)

    return tasksets
