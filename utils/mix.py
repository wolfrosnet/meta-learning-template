import torch
from torch.distributions import Beta
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam.cpu())
    cut_w = np.int16(W * cut_rat)
    cut_h = np.int16(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = int(np.clip(cx - cut_w // 2, 0, W))
    bby1 = int(np.clip(cy - cut_h // 2, 0, H))
    bbx2 = int(np.clip(cx + cut_w // 2, 0, W))
    bby2 = int(np.clip(cy + cut_h // 2, 0, H))

    return bbx1, bby1, bbx2, bby2


def cutmix_data(data1, data2, lam):
    mixed_data = data2.clone()

    bbx1, bby1, bbx2, bby2 = rand_bbox(data2.size(), lam)

    mixed_data[:, :, bbx1:bbx2, bby1:bby2] = data1[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data2.size()[-1] * data2.size()[-2]))

    return mixed_data


def mixup_data(data1, data2, lam):
    mixed_data = data1*lam + data2*(1-lam)

    return mixed_data


def crosstask_mixup(batch1, batch2, device, mode='cutmix', inner=False, return_lambda=False):
    lam = Beta(torch.FloatTensor([2]), torch.FloatTensor([2])).sample().to("cuda")

    data1, label1 = batch1
    data1 = data1.to(device)
    label1 = label1.to(device)
    sort = torch.sort(label1)
    data1 = data1.squeeze(0)[sort.indices].squeeze(0)
    label1 = label1.squeeze(0)[sort.indices].squeeze(0)

    data2, label2 = batch2
    data2 = data2.to(device)
    label2 = label2.to(device)

    if inner:
        sort = torch.sort(label2, descending=True)
    else:
        sort = torch.sort(label2)

    data2 = data2.squeeze(0)[sort.indices].squeeze(0)
    label2 = label2.squeeze(0)[sort.indices].squeeze(0)

    if mode == 'cutmix':
        mixed_data = cutmix_data(data1, data2, lam)
    elif mode == 'mixup':
        mixed_data = mixup_data(data1, data2, lam)
    else:
        raise NotImplementedError

    if return_lambda:
        return [mixed_data, label1], lam
    else:
        return [mixed_data, label1]
