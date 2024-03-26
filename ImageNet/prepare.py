import torch
import torchvision

import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./')
args = parser.parse_args()

if not os.path.exists(args.path+'Models'):
    os.makedirs(args.path+'Models')
    
    resnet50_1 = torchvision.models.resnet50()
    torch.save(resnet50_1.state_dict(), args.path+'Models/ResNet50_1.pt')
    
    resnet50_2 = torchvision.models.resnet50()
    torch.save(resnet50_2.state_dict(), args.path+'Models/ResNet50_2.pt')

    resnet50_3 = torchvision.models.resnet50()
    torch.save(resnet50_3.state_dict(), args.path+'Models/ResNet50_3.pt')

    '''
    resnet50 = torchvision.models.resnet50()
    torch.save(resnet50.state_dict(), args.path+'Models/ResNet50_1.pt')
    torch.save(resnet50.state_dict(), args.path+'Models/ResNet50_2.pt')
    torch.save(resnet50.state_dict(), args.path+'Models/ResNet50_3.pt')
    '''

if not os.path.exists(args.path+'Saved_Models'):
    os.makedirs(args.path+'Saved_Models')

if not os.path.exists(args.path+'Results'):
    os.makedirs(args.path+'Results')
    if not os.path.exists(args.path+'Results/Presentation'):
        os.makedirs(args.path+'Results/Presentation')
    if not os.path.exists(args.path+'Results/ForPlotting'):
        os.makedirs(args.path+'Results/ForPlotting')