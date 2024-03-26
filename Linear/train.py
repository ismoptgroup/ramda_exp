import torch
import torchvision
import argparse
import os
import sys 
sys.path.append("..")

from Core.group import group_model
from Core.optimizer import ProxSGD, ProxAdamW, RMDA, RAMDA, AdamW
from Core.scheduler import multistep_param_scheduler

from model import Lin

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--optimizer', type=str, default='RMDA')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--momentum', type=float, default=1e-2)
parser.add_argument('--lambda_', type=float, default=1e-3)
parser.add_argument('--milestones', type=int, nargs='+', default=[i for i in range(100, 500, 100)])
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
training_dataset = torchvision.datasets.MNIST(root=args.path+'Data',
                                              train=True,
                                              download=False,
                                              transform=transforms)
training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                                  batch_size=args.batch_size, 
                                                  shuffle=True,
                                                  num_workers=args.num_workers)
testing_dataset = torchvision.datasets.MNIST(root=args.path+'Data', 
                                             train=False, 
                                             download=False,
                                             transform=transforms)
testing_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset, 
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers)
    
model = Lin()
model.load_state_dict(torch.load(args.path+'Models/Lin_MNIST_'+str(args.seed)+'.pt'))

criterion = torch.nn.NLLLoss()

optimizer_grouped_parameters = group_model(model=model, name="Lin", lambda_=args.lambda_)
if args.optimizer != "ProxSSI":
    if args.optimizer == "ProxSGD":
        optimizer = ProxSGD(params=optimizer_grouped_parameters,
                            lr=args.lr)
    elif args.optimizer == "ProxAdamW":
        optimizer = ProxAdamW(params=optimizer_grouped_parameters,
                              lr=args.lr)
    elif args.optimizer == "RMDA":
        optimizer = RMDA(params=optimizer_grouped_parameters,
                         lr=args.lr,
                         momentum=args.momentum)
    elif args.optimizer == "RAMDA":
        optimizer = RAMDA(params=optimizer_grouped_parameters,
                          lr=args.lr)
else:
    # https://github.com/tristandeleu/pytorch-structured-sparsity/blob/master/proxssi/groups/resnet.py
    def conv_groups_fn(params):
        return [param.view(param.shape[:2] + (-1,)).transpose(2, 1) for param in params]

    def column_groups_fn(params):
        return [param.unsqueeze(0) for param in params]

    def groups(model):
        to_prox_conv, to_prox_linear = [], []
        remaining = []

        for name, param in model.named_parameters():
            if 'weight' in name:
                if param.ndim == 4:
                    to_prox_conv.append(param)
                elif param.ndim == 2:
                    to_prox_linear.append(param)
                else:
                    remaining.append(param)
            else:
                remaining.append(param)

        optimizer_grouped_parameters = [
            {
                'params': to_prox_conv,
                'weight_decay': 0.0,
                'groups_fn': conv_groups_fn
            },
            {
                'params': to_prox_linear,
                'weight_decay': 0.0,
                'groups_fn': column_groups_fn
            }
        ]
        if remaining:
            optimizer_grouped_parameters.append({
                'params': remaining,
                'weight_decay': 0.0,
                'groups_fn': None
            })

        return optimizer_grouped_parameters
    
    grouped_params = groups(model)
    
    assert grouped_params[0]["params"] == []
    assert optimizer_grouped_parameters[0]["params"] == grouped_params[1]["params"]
    assert optimizer_grouped_parameters[1]["params"] == grouped_params[2]["params"]

    prox_kwargs = {
        'lr': args.lr,
        'lambda_': args.lambda_
    }
    optimizer_kwards = {
        'lr': args.lr,
        'penalty': 'l1_l2',
        'weight_decay': 0.0,
        'prox_kwargs': prox_kwargs
    }
    
    optimizer = AdamW(grouped_params, **optimizer_kwards)    
    
scheduler = multistep_param_scheduler(name=args.optimizer, optimizer=optimizer, milestones=args.milestones, gamma=args.gamma) 

lrs = []
momentums = []
training_objectives = []
training_accuracies = []
validation_accuracies = []
unstructured_sparsities = []
structured_sparsities = []
weighted_structured_sparsities = []
    
for epoch in range(args.epochs):
    model.train()
    for X, y in training_dataloader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        scheduler.momentum_step(optimizer=optimizer, epoch=epoch)
    
    model.eval()
    for param_group in optimizer.param_groups:
        lr = param_group['lr'] 
        if args.optimizer == "MSGD" or args.optimizer == "ProxSGD" or args.optimizer == "RMDA" or args.optimizer == "RAMDA":
            momentum = param_group['momentum']
        elif args.optimizer == "ProxAdamW" or args.optimizer == "ProxSSI":
            momentum = param_group['betas']
            
    training_objective = 0.0
    training_accuracy = 0.0
    with torch.no_grad():
        for X, y in training_dataloader:
            y_hat = model(X)
            loss = criterion(y_hat, y)
            y_hat = y_hat.argmax(dim=1)
            training_objective += loss.item()*len(y)
            training_accuracy += y_hat.eq(y.view_as(y_hat)).float().sum().item()

    training_objective /= len(training_dataset)
    training_accuracy /= len(training_dataset)
    
    for group in optimizer_grouped_parameters:
        dim = group["dim"]
        lambda_ = group["lambda_"]
        for p in group["params"]:
            if dim == (0):
                reg_scaling = p.shape[0]**0.5
                training_objective += reg_scaling*lambda_*(torch.linalg.norm(p, dim=dim).sum().item()) 
                
    validation_accuracy = 0.0
    with torch.no_grad():
        for X, y in testing_dataloader:
            output = model(X)
            y_hat = output.argmax(dim=1)
            validation_accuracy += y_hat.eq(y.view_as(y_hat)).float().sum().item()
        validation_accuracy /= len(testing_dataset)
                
    # unstructured sparsity            
    nonzero = 0.0
    num_el = 0.0
    for p in model.parameters():
        nonzero += p.count_nonzero().item()
        num_el += p.numel()
    unstructured_sparsity = 1.0-(nonzero/num_el)
    
    # (input-wise) structured sparsity  
    num_nonsparse_groups = 0.0
    num_groups = 0.0
    for group in optimizer_grouped_parameters:
        dim = group["dim"]
        for p in group["params"]:
            if dim == (0):
                num_nonsparse_groups += p.count_nonzero(dim=dim).count_nonzero().item()
                num_groups += p.shape[1]
    structured_sparsity = 1.0-(num_nonsparse_groups/num_groups)
        
    # weighted (input-wise) structured sparsity
    weighted_num_nonsparse_groups = 0.0
    weighted_num_groups = 0.0
    for group in optimizer_grouped_parameters:
        dim = group["dim"]
        for p in group["params"]:
            if dim == (0):
                group_size = p.shape[0]
                weighted_num_nonsparse_groups += p.count_nonzero(dim=dim).count_nonzero().item()*group_size
                weighted_num_groups += p.shape[1]*group_size
    weighted_structured_sparsity = 1.0-(weighted_num_nonsparse_groups/weighted_num_groups)
    
    lrs.append(lr)
    momentums.append(momentum)
    training_objectives.append(training_objective)
    training_accuracies.append(training_accuracy)
    validation_accuracies.append(validation_accuracy)
    unstructured_sparsities.append(unstructured_sparsity)
    structured_sparsities.append(structured_sparsity)
    weighted_structured_sparsities.append(weighted_structured_sparsity)
    
    print("optimizer: {}".format(args.optimizer))
    print("epochs: {}".format(epoch+1))
    print("learning rate: {}".format(lr))
    print("momentum: {}".format(momentum))
    print("training objective: {}".format(training_objective))
    print("training accuracy: {}".format(training_accuracy))
    print("validation accuracy: {}".format(validation_accuracy))
    print("unstructured sparsity: {}".format(unstructured_sparsity))
    print("structured sparsity: {}".format(structured_sparsity))
    print("weighted structured sparsity: {}".format(weighted_structured_sparsity))
        
    f = open(args.path+'Results/Presentation/'+args.optimizer+'_Lin_on_MNIST_presentation_'+str(args.seed)+'.txt', 'w+')  

    f.write("final training objective: {}".format(training_objective)+'\n')
    f.write("final training accuracy: {}".format(training_accuracy)+'\n')
    f.write("final validation accuracy: {}".format(validation_accuracy)+'\n')
    f.write("final unstructured sparsity: {}".format(unstructured_sparsity)+'\n')
    f.write("final structured sparsity: {}".format(structured_sparsity)+'\n')
    f.write("final weighted structured sparsity: {}".format(weighted_structured_sparsity)+'\n')    

    f.write("batch size: {}".format(args.batch_size)+'\n')
    f.write("num workers: {}".format(args.num_workers)+'\n')
    f.write("optimizer: {}".format(args.optimizer)+'\n')
    f.write("epochs: {}".format(args.epochs)+'\n')
    f.write("lr: {}".format(args.lr)+'\n')
    f.write("lambda_: {}".format(args.lambda_)+'\n')
    f.write("milestones: {}".format(args.milestones)+'\n')
    f.write("gamma: {}".format(args.gamma)+'\n')
    f.write("path: {}".format(args.path)+'\n')
    f.write("seed: {}".format(args.seed)+'\n')

    if args.optimizer == "MSGD" or args.optimizer == "ProxSGD" or args.optimizer == "RMDA" or args.optimizer == "RAMDA":
        for i, r in enumerate(zip(lrs, momentums, training_objectives, training_accuracies, validation_accuracies, unstructured_sparsities, structured_sparsities, weighted_structured_sparsities)):
            f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f}\ttraining objective:{:<20.15f}\ttraining accuracy:{:<20.15f}\tvalidation accuracy:{:<20.15f}\tunstructured sparsity:{:<20.15f}\tstructured sparsity:{:<20.15f}\tweighted structured sparsity:{:<20.15f}".format((i+1), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])+'\n')
            
    elif args.optimizer == "ProxAdamW" or args.optimizer == "ProxSSI":
        for i, r in enumerate(zip(lrs, momentums, training_objectives, training_accuracies, validation_accuracies, unstructured_sparsities, structured_sparsities, weighted_structured_sparsities)):
            f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f},{:<20.15f}\ttraining objective:{:<20.15f}\ttraining accuracy:{:<20.15f}\tvalidation accuracy:{:<20.15f}\tunstructured sparsity:{:<20.15f}\tstructured sparsity:{:<20.15f}\tweighted structured sparsity:{:<20.15f}".format((i+1), r[0], r[1][0], r[1][1], r[2], r[3], r[4], r[5], r[6], r[7])+'\n')    

    f.close()
    
    f = open(args.path+'Results/ForPlotting/'+args.optimizer+'_Lin_on_MNIST_forplotting_'+str(args.seed)+'.txt', 'w+')

    f.write('learning rate\n')
    for i, r in enumerate(lrs):
         f.write("epoch {}: {}".format((i+1), r)+'\n')
    
    f.write('momentum\n')
    for i, r in enumerate(momentums):
         f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.write('training objective\n')
    for i, r in enumerate(training_objectives):
         f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.write('training accuracy\n')
    for i, r in enumerate(training_accuracies):
         f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.write('validation accuracy\n')
    for i, r in enumerate(validation_accuracies):
        f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.write('unstructured sparsity\n')
    for i, r in enumerate(unstructured_sparsities):
        f.write("epoch {}: {}".format((i+1), r)+'\n')
        
    f.write('structured sparsity\n')
    for i, r in enumerate(structured_sparsities):
        f.write("epoch {}: {}".format((i+1), r)+'\n')
    
    f.write('weighted structured sparsity\n')
    for i, r in enumerate(weighted_structured_sparsities):
        f.write("epoch {}: {}".format((i+1), r)+'\n')

    f.close()
        
    torch.save(model.state_dict(), args.path+'Saved_Models/'+args.optimizer+'_Lin_on_MNIST_'+str(args.seed)+'.pt')
     
    # schedule learning rate and (restart for RMDA and RAMDA)
    scheduler.step(optimizer=optimizer, epoch=epoch)