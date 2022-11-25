import argparse


def get_args():
    parser = argparse.ArgumentParser("SNASNet")
    parser.add_argument('--exp_name', type=str, default='snn_test', help='experiment name')
    parser.add_argument('--data_dir', type=str, default='dataset/', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100]')
    parser.add_argument('--seed', default=1234, type=int)

    parser.add_argument('--timestep', type=int, default=5, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--tau', type=float, default=4 / 3, help='neuron decay time factor')
    parser.add_argument('--threshold', type=float, default=1.0, help='neuron firing threshold')

    parser.add_argument('--arch', type=str, default='fc2', help='[fc2]')
    parser.add_argument('--optimizer', type=str, default='adam', help='[sgd, adam]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine]')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='learnng rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument('--valid_freq', type=int, default=20, help='test for SNN')
    parser.add_argument("--print_freq", default=20, type=int)

    parser.add_argument('--noisetype', type=str, default='blur', help='[blur, gaussian_noise]')
    parser.add_argument('--noisetime',  nargs='+', type=int, default=False)
    parser.add_argument('--lossskiptime',  nargs='+', type=int, default=False)
    parser.add_argument('--freqlevel', type=str, default='high', help='[high, low]')
    parser.add_argument("--datanum", default=50000, type=int)
    parser.add_argument("--label_corrupt_prob", default=1.0, type=float)
    parser.add_argument("--layernum", default=2, type=float)

    parser.add_argument("--loss_threshold", default=0, type=float)
    parser.add_argument('--attack_method', type=str, default="pgd", help="[fgsm, pgd]")
    parser.add_argument("--encode", default="d", type=str, help="Encoding [p d]")

    args = parser.parse_args()
    print(args)

    return args
