import os
import time
import torch
import numpy as np
import torchvision.transforms as transforms
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)

        res.append(correct_k.mul_(100.0 / batch_size))
    return res





def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./snapshots/"+tag):
        os.makedirs("./snapshots/"+tag)
    filename = os.path.join("./snapshots/"+tag+"/{}_ckpt_{:04}.pth.tar".format(tag, iters))
    torch.save(state, filename)



def data_transforms(args):

    MEAN = [0.4913, 0.4821, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return train_transform, valid_transform



def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    print('Elapsed time: hour: %d, minute: %d, second: %f' % (hour, minute, second))




def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
