import torch
import torchvision

import argparse
import os

from model import VGG19, ResNet50 

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./')
args = parser.parse_args()

if not os.path.exists(args.path+'Models'):
    os.makedirs(args.path+'Models')
    
    vgg19_cifar10_1 = VGG19(num_classes=10)
    torch.save(vgg19_cifar10_1.state_dict(), args.path+'Models/VGG19_CIFAR10_1.pt')

    vgg19_cifar10_2 = VGG19(num_classes=10)
    torch.save(vgg19_cifar10_2.state_dict(), args.path+'Models/VGG19_CIFAR10_2.pt')

    vgg19_cifar10_3 = VGG19(num_classes=10)
    torch.save(vgg19_cifar10_3.state_dict(), args.path+'Models/VGG19_CIFAR10_3.pt')

    vgg19_cifar100_1 = VGG19(num_classes=100)
    torch.save(vgg19_cifar100_1.state_dict(), args.path+'Models/VGG19_CIFAR100_1.pt')

    vgg19_cifar100_2 = VGG19(num_classes=100)
    torch.save(vgg19_cifar100_2.state_dict(), args.path+'Models/VGG19_CIFAR100_2.pt')
    
    vgg19_cifar100_3 = VGG19(num_classes=100)
    torch.save(vgg19_cifar100_3.state_dict(), args.path+'Models/VGG19_CIFAR100_3.pt')

    resnet50_cifar10_1 = ResNet50(num_classes=10)
    torch.save(resnet50_cifar10_1.state_dict(), args.path+'Models/ResNet50_CIFAR10_1.pt')

    resnet50_cifar10_2 = ResNet50(num_classes=10)
    torch.save(resnet50_cifar10_2.state_dict(), args.path+'Models/ResNet50_CIFAR10_2.pt')

    resnet50_cifar10_3 = ResNet50(num_classes=10)
    torch.save(resnet50_cifar10_3.state_dict(), args.path+'Models/ResNet50_CIFAR10_3.pt')

    resnet50_cifar100_1 = ResNet50(num_classes=100)
    torch.save(resnet50_cifar100_1.state_dict(), args.path+'Models/ResNet50_CIFAR100_1.pt')

    resnet50_cifar100_2 = ResNet50(num_classes=100)
    torch.save(resnet50_cifar100_2.state_dict(), args.path+'Models/ResNet50_CIFAR100_2.pt')

    resnet50_cifar100_3 = ResNet50(num_classes=100)
    torch.save(resnet50_cifar100_3.state_dict(), args.path+'Models/ResNet50_CIFAR100_3.pt')

torchvision.datasets.CIFAR10(root=args.path+'Data', train=True, download=True)
torchvision.datasets.CIFAR10(root=args.path+'Data', train=False, download=True)

torchvision.datasets.CIFAR100(root=args.path+'Data', train=True, download=True)
torchvision.datasets.CIFAR100(root=args.path+'Data', train=False, download=True)

if not os.path.exists(args.path+'Saved_Models'):
    os.makedirs(args.path+'Saved_Models')

if not os.path.exists(args.path+'Results'):
    os.makedirs(args.path+'Results')
    if not os.path.exists(args.path+'Results/Presentation'):
        os.makedirs(args.path+'Results/Presentation')
    if not os.path.exists(args.path+'Results/ForPlotting'):
        os.makedirs(args.path+'Results/ForPlotting')