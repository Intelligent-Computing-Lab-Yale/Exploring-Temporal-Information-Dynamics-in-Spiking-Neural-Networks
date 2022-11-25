import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn

from archs.resnet_snn import ResNet19
from torch.utils.data import Dataset
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net



def main():
    args = config.get_args()

    # define dataset
    train_transform, valid_transform = data_transforms(args)

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, pin_memory=True, num_workers=4)
    valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False,
                                          download=True, transform=valid_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, pin_memory=True, num_workers=4)
    n_class = 10




    # define model
    model = ResNet19(num_classes=n_class, total_timestep=args.timestep).cuda()


    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 0)


    start = time.time()
    for epoch in range(args.epochs):
        train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
        scheduler.step()
        if (epoch + 1) % args.valid_freq == 0:
            validate(args, epoch, val_loader, model, criterion)
            utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1, tag='T'+str(args.timestep)+'_D'+str(args.dataset)+'_A'+str(args.arch)+'_ce')

    utils.time_record(start)


def train(args, epoch, train_data,  model, criterion, optimizer, scheduler):
    model.train()

    top1 = utils.AvgrageMeter()
    train_loss = 0.0

    if (epoch + 1) % 10 == 0:
        print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_lr()[0]))

    for step, (inputs, targets) in enumerate(train_data):

        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        output_list = model(inputs)
        loss = criterion(sum(output_list)/ args.timestep, targets)
        loss.backward()

        prec1, prec5 = utils.accuracy(sum(output_list), targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()

        reset_net(model)
        optimizer.step()


    if (epoch + 1) % 10 == 0:
        print('train_loss: %.6f' % (train_loss / len(train_data)), 'train_acc: %.6f' % top1.avg)


def validate(args, epoch, val_data, model, criterion):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            loss = criterion(sum(outputs), targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(sum(outputs), targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)

            reset_net(model)

        print('[Val_Accuracy epoch:%d] val_acc:%f'
              % (epoch + 1,  val_top1.avg))
        return val_top1.avg


if __name__ == '__main__':
    main()
