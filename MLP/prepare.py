import torch
import torchvision

import argparse
import os

from model import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./')
args = parser.parse_args()

if not os.path.exists(args.path+'Models'):
    os.makedirs(args.path+'Models')
    
    mlp_fashiomnist_1 = MLP()
    mlp_fashiomnist_2 = MLP()
    mlp_fashiomnist_3 = MLP()
    torch.save(mlp_fashiomnist_1.state_dict(), args.path+'Models/MLP_FashionMNIST_1.pt')
    torch.save(mlp_fashiomnist_2.state_dict(), args.path+'Models/MLP_FashionMNIST_2.pt')
    torch.save(mlp_fashiomnist_3.state_dict(), args.path+'Models/MLP_FashionMNIST_3.pt')

torchvision.datasets.FashionMNIST(root=args.path+'Data', train=True, download=True)
torchvision.datasets.FashionMNIST(root=args.path+'Data', train=False, download=True)
    
if not os.path.exists(args.path+'Saved_Models'):
    os.makedirs(args.path+'Saved_Models')

if not os.path.exists(args.path+'Results'):
    os.makedirs(args.path+'Results')
    if not os.path.exists(args.path+'Results/Presentation'):
        os.makedirs(args.path+'Results/Presentation')
    if not os.path.exists(args.path+'Results/ForPlotting'):
        os.makedirs(args.path+'Results/ForPlotting')