import os
import torch
import utils
import config
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from archs.resnet_snn import ResNet19
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net
from copy import deepcopy

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


    fisherlist = []
    for t in range(args.timestep):
        fisherlist.append([])

    epochlist= [20,120,300]


    ep_ic_list = []
    for ep in epochlist:

        #TODO need to load model .pth here
        model.load_state_dict(torch.load(f"snapshots/T{str(args.timestep)}_D{str(args.dataset)}_A{str(args.arch)}_ce/T{str(args.timestep)}_D{str(args.dataset)}_A{str(args.arch)}_ce_ckpt_{str(ep).zfill(4)}.pth.tar")['state_dict'])

        print ('Ep',str(ep),'--total time', str(args.timestep))

        ep_fisher_list = []
        for timestep in range(1, args.timestep+1):
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}

            precision_matrices = {}
            for n, p in deepcopy(params).items():
                p.data.zero_()
                precision_matrices[n] = p.data
            model.eval()

            for step, (input, target) in enumerate(train_loader):
                model.zero_grad()
                input = input.cuda()
                target = target.cuda()

                output = sum(model(input)[:timestep])/timestep

                loss = F.nll_loss(F.log_softmax(output, dim=1), target)
                loss.backward()

                for n, p in model.named_parameters():
                    precision_matrices[n].data += p.grad.data ** 2 /100#len(train_loader)
                reset_net(model)

                if step == 100:
                    break


            precision_matrices = {n: p for n, p in precision_matrices.items()}

            fisher_trace_info  = 0
            for p in precision_matrices:
                weight = precision_matrices[p]
                fisher_trace_info += weight.sum()

            print ("time", timestep, fisher_trace_info)
            fisherlist[timestep-1].append(float(fisher_trace_info.cpu().data.numpy()))
            ep_fisher_list.append(float(fisher_trace_info.cpu().data.numpy()))


        print ('fisher list', ep_fisher_list)


    fisher_print = []
    for t in range(args.timestep):
        print ("----------fisher info at time", t)
        print (fisherlist[t])
        fisher_print.append(fisherlist[t][0])
    print ("fisher_print", fisher_print)



if __name__ == '__main__':
    main()
