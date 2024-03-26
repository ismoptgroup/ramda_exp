import torch
import torchvision
import argparse
import os
import time
import sys 
sys.path.append("..")

from Core.group import group_model
from Core.optimizer import ProxSGD, ProxAdamW, RMDA, RAMDA
from Core.scheduler import multistep_param_scheduler

from model import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--tuning', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--optimizer', type=str, default='RAMDA')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--momentum', type=float, default=1e-1)
parser.add_argument('--lambda_', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--regularization', type=str, default='nuclear')
parser.add_argument('--rtol', type=float, default=1e-3)
parser.add_argument('--max-iters', type=int, default=5)
parser.add_argument('--milestones', type=int, nargs='+', default=[i for i in range(100, 300, 100)])
parser.add_argument('--gamma', type=float, default=1e-1)
parser.add_argument('--path', type=str, default='./')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

torch.set_num_threads(args.num_workers)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                             torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])
training_dataset = torchvision.datasets.FashionMNIST(root=args.path+'Data',
                                                     train=True,
                                                     download=False,
                                                     transform=transforms)
training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                                  batch_size=args.batch_size, 
                                                  shuffle=True,
                                                  num_workers=args.num_workers)
testing_dataset = torchvision.datasets.FashionMNIST(root=args.path+'Data', 
                                                    train=False, 
                                                    download=False,
                                                    transform=transforms)
testing_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset, 
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers)
    
model = MLP()
model.load_state_dict(torch.load(args.path+'Models/MLP_FashionMNIST_'+str(args.seed)+'.pt'))
model.cuda()

criterion = torch.nn.NLLLoss()

optimizer_grouped_parameters = group_model(model=model, name="MLP", lambda_=args.lambda_)
if args.optimizer == "ProxSGD":
    optimizer = ProxSGD(params=optimizer_grouped_parameters,
                        lr=args.lr,
                        regularization=args.regularization)
elif args.optimizer == "ProxAdamW":
    optimizer = ProxAdamW(params=optimizer_grouped_parameters,
                          lr=args.lr,
                          regularization=args.regularization,
                          rtol=args.rtol,
                          max_iters=args.max_iters)
elif args.optimizer == "RMDA":
    optimizer = RMDA(params=optimizer_grouped_parameters,
                     lr=args.lr,
                     momentum=args.momentum,
                     regularization=args.regularization)
elif args.optimizer == "RAMDA":
    optimizer = RAMDA(params=optimizer_grouped_parameters,
                      lr=args.lr,
                      momentum=args.momentum,
                      regularization=args.regularization,
                      rtol=args.rtol,
                      max_iters=args.max_iters)   
elif args.optimizer == "MSGD":
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)     
scheduler = multistep_param_scheduler(name=args.optimizer, optimizer=optimizer, milestones=args.milestones, gamma=args.gamma) 

lrs = []
momentums = []
training_losses = []
training_accuracies = []
validation_accuracies = []
low_rank_levels = []
epoch_times = []

for epoch in range(args.epochs):
    start = time.time()
    model.train()
    training_loss = 0.0
    training_accuracy = 0.0
    for X, y in training_dataloader:
        X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        scheduler.momentum_step(optimizer=optimizer, epoch=epoch)
        training_loss += loss.item()*len(y)
        y_hat = output.argmax(dim=1)
        training_accuracy += y_hat.eq(y.view_as(y_hat)).float().sum().item()
    end = time.time()

    training_loss /= len(training_dataset)
    training_accuracy /= len(training_dataset)
    
    model.eval()
    for param_group in optimizer.param_groups:
        lr = param_group['lr'] 
        if args.optimizer == "MSGD" or args.optimizer == "ProxSGD" or args.optimizer == "RMDA" or args.optimizer == "RAMDA":
            momentum = param_group['momentum']
        elif args.optimizer == "ProxAdamW":
            momentum = param_group['betas']
                
    validation_accuracy = 0.0
    with torch.no_grad():
        for X, y in testing_dataloader:
            X, y = X.cuda(), y.cuda()
            output = model(X)
            y_hat = output.argmax(dim=1)
            validation_accuracy += y_hat.eq(y.view_as(y_hat)).float().sum().item()
        validation_accuracy /= len(testing_dataset)

    if args.optimizer != "MSGD":
        S_hats = []
        for group in optimizer_grouped_parameters:
            dim = group["dim"]
            lambda_ = group["lambda_"]
            if dim == (0):
                for p in group["params"]:
                    S_hats.append(optimizer.state[p]['S'])
     
        nonzero = 0.0
        num_el = 0.0
        for S_hat in S_hats:
            nonzero += S_hat.count_nonzero().item()
            num_el += S_hat.numel()
        low_rank_level = 1.0-(nonzero/num_el)
    else:
        low_rank_level = 0.0

    epoch_time = end-start
        
    lrs.append(lr)
    momentums.append(momentum)
    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)
    validation_accuracies.append(validation_accuracy)
    low_rank_levels.append(low_rank_level)
    epoch_times.append(epoch_time)
    
    print("optimizer: {}".format(args.optimizer))
    print("epochs: {}".format(epoch+1))
    print("learning rate: {}".format(lr))
    print("momentum: {}".format(momentum))
    print("training loss: {}".format(training_loss))
    print("training accuracy: {}".format(training_accuracy))
    print("validation accuracy: {}".format(validation_accuracy))
    print("low rank level: {}".format(low_rank_level))
    print("epoch time: {}".format(epoch_time))

    if args.tuning:
        f = open(args.path+'Results/Presentation/'+args.optimizer+'_MLP_on_FashionMNIST_presentation_'+str(args.lr)+'_'+str(args.lambda_)+'_'+str(args.seed)+'.txt', 'w+') 
    else:
        f = open(args.path+'Results/Presentation/'+args.optimizer+'_MLP_on_FashionMNIST_presentation_'+str(args.seed)+'.txt', 'w+')  
    
    f.write("final training loss: {}".format(training_loss)+'\n')
    f.write("final training accuracy: {}".format(training_accuracy)+'\n')
    f.write("final validation accuracy: {}".format(validation_accuracy)+'\n')
    f.write("final low rank level: {}".format(low_rank_level)+'\n') 

    f.write("batch size: {}".format(args.batch_size)+'\n')
    f.write("num workers: {}".format(args.num_workers)+'\n')
    f.write("optimizer: {}".format(args.optimizer)+'\n')
    f.write("epochs: {}".format(args.epochs)+'\n')
    f.write("lr: {}".format(args.lr)+'\n')
    f.write("lambda_: {}".format(args.lambda_)+'\n')
    f.write("weight decay: {}".format(args.weight_decay)+'\n')
    f.write("regularization: {}".format(args.regularization)+'\n')
    f.write("rtol: {}".format(args.rtol)+'\n')
    f.write("max iters: {}".format(args.max_iters)+'\n')
    f.write("milestones: {}".format(args.milestones)+'\n')
    f.write("gamma: {}".format(args.gamma)+'\n')
    f.write("path: {}".format(args.path)+'\n')
    f.write("seed: {}".format(args.seed)+'\n')

    if args.optimizer == "MSGD" or args.optimizer == "ProxSGD" or args.optimizer == "RMDA" or args.optimizer == "RAMDA":
        for i, r in enumerate(zip(lrs, momentums, training_losses, training_accuracies, validation_accuracies, low_rank_levels, epoch_times)):
            f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f}\ttraining loss:{:<20.15f}\ttraining accuracy:{:<20.15f}\tvalidation accuracy:{:<20.15f}\tlow rank level:{:<20.15f}\tepoch time:{:<20.15f}".format((i+1), r[0], r[1], r[2], r[3], r[4], r[5], r[6])+'\n')
            
    elif args.optimizer == "ProxAdamW":
        for i, r in enumerate(zip(lrs, momentums, training_losses, training_accuracies, validation_accuracies, low_rank_levels, epoch_times)):
            f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f},{:<20.15f}\ttraining loss:{:<20.15f}\ttraining accuracy:{:<20.15f}\tvalidation accuracy:{:<20.15f}\tlow rank level:{:<20.15f}\tepoch time:{:<20.15f}".format((i+1), r[0], r[1][0], r[1][1], r[2], r[3], r[4], r[5], r[6])+'\n')    

    f.close()
    
    if args.tuning:
        f = open(args.path+'Results/ForPlotting/'+args.optimizer+'_MLP_on_FashionMNIST_forplotting_'+str(args.lr)+'_'+str(args.lambda_)+'_'+str(args.seed)+'.txt', 'w+')
    else:
        f = open(args.path+'Results/ForPlotting/'+args.optimizer+'_MLP_on_FashionMNIST_forplotting_'+str(args.seed)+'.txt', 'w+')
        
    f.write('learning rate\n')
    for i, r in enumerate(lrs):
         f.write("epoch {}: {}".format((i+1), r)+'\n')
    
    f.write('momentum\n')
    for i, r in enumerate(momentums):
         f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.write('training loss\n')
    for i, r in enumerate(training_losses):
         f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.write('training accuracy\n')
    for i, r in enumerate(training_accuracies):
         f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.write('validation accuracy\n')
    for i, r in enumerate(validation_accuracies):
        f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.write('low rank level\n')
    for i, r in enumerate(low_rank_levels):
        f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.write('epoch time\n')
    for i, r in enumerate(epoch_times):
        f.write("epoch {}: {}".format((i+1), r)+'\n')
        
    f.close()
        
    torch.save(model.state_dict(), args.path+'Saved_Models/'+args.optimizer+'_MLP_on_FashionMNIST_'+str(args.seed)+'.pt')
     
    # schedule learning rate and (restart for RMDA and RAMDA)
    scheduler.step(optimizer=optimizer, epoch=epoch)