import torch
import torchvision
import argparse
import os

from Core.group import group_model
from Core.optimizer import ProxAdamW, RAMDA
from Core.scheduler import multistep_param_scheduler
from Core.prox_fns import prox_glasso

from model import Lin, VGG19, ResNet50

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ResNet50")
parser.add_argument('--dataset', type=str, default="CIFAR100")
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--optimizer', type=str, default='RAMDA')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--lambda_', type=float, default=0.0)
parser.add_argument('--max-iters', type=int, default=100)
parser.add_argument('--early-stopping', action='store_true', default=False)
parser.add_argument('--milestones', type=int, nargs='+', default=[i for i in range(200, 1000, 200)])
parser.add_argument('--gamma', type=float, default=1e-1)
parser.add_argument('--path', type=str, default='./')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

torch.set_num_threads(args.num_workers)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

assert (args.model == "Lin" and args.dataset == "MNIST") or (args.model == "VGG19" and (args.dataset == "CIFAR10" or args.dataset == "CIFAR100")) or (args.model == "ResNet50" and (args.dataset == "CIFAR10" or args.dataset == "CIFAR100"))

if args.dataset == "MNIST":
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
elif args.dataset == "CIFAR10":
    training_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testing_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    training_dataset = torchvision.datasets.CIFAR10(root=args.path+'Data',
                                                    train=True,
                                                    download=False,
                                                    transform=training_transforms)
    training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.num_workers,
                                                      pin_memory=True)
    testing_dataset = torchvision.datasets.CIFAR10(root=args.path+'Data', 
                                                   train=False,
                                                   download=False,
                                                   transform=testing_transforms)
    testing_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=args.num_workers,
                                                     pin_memory=True)
elif args.dataset == "CIFAR100":
    training_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    testing_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    training_dataset = torchvision.datasets.CIFAR100(root=args.path+'Data',
                                                     train=True,
                                                     download=False,
                                                     transform=training_transforms)
    training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.num_workers,
                                                      pin_memory=True)
    testing_dataset = torchvision.datasets.CIFAR100(root=args.path+'Data', 
                                                    train=False,
                                                    download=False,
                                                    transform=testing_transforms)
    testing_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=args.num_workers,
                                                     pin_memory=True)

if args.model == "Lin" and args.dataset == "MNIST":
    model = Lin()
elif args.model == "VGG19" and args.dataset == "CIFAR10":
    model = VGG19(num_classes=10)
elif args.model == "VGG19" and args.dataset == "CIFAR100":
    model = VGG19(num_classes=100)
elif args.model == "ResNet50" and args.dataset == "CIFAR10":
    model = ResNet50(num_classes=10)
elif args.model == "ResNet50" and args.dataset == "CIFAR100":
    model = ResNet50(num_classes=100)
model.load_state_dict(torch.load(args.path+'Models/'+args.model+'_'+args.dataset+'_'+str(args.seed)+'.pt'))
if args.model != "Lin":
    model.cuda()

if args.model == "Lin":
    optimizer_grouped_parameters = group_model(model=model, name="Lin", lambda_=args.lambda_)
elif args.model == "VGG19":
    optimizer_grouped_parameters = group_model(model=model, name="VGG", lambda_=args.lambda_)
elif args.model == "ResNet50":
    optimizer_grouped_parameters = group_model(model=model, name="ResNet", lambda_=args.lambda_)

if args.model != "Lin":
    criterion = torch.nn.NLLLoss().cuda()
else:
    criterion = torch.nn.NLLLoss()

if args.optimizer == "ProxAdamW":
    optimizer = ProxAdamW(params=optimizer_grouped_parameters,
                          lr=args.lr,
                          max_iters=args.max_iters,
                          early_stopping=args.early_stopping)
elif args.optimizer == "RAMDA":
    optimizer = RAMDA(params=optimizer_grouped_parameters,
                      lr=args.lr,
                      max_iters=args.max_iters,
                      early_stopping=args.early_stopping)
    
scheduler = multistep_param_scheduler(name=args.optimizer, optimizer=optimizer, milestones=args.milestones, gamma=args.gamma) 

lrs = []
momentums = []
training_objectives = []
training_accuracies = []
validation_accuracies = []
unstructured_sparsities = []
structured_sparsities = []
weighted_structured_sparsities = []

prox_grad_norms = []
iterations = []
times = []

for epoch in range(args.epochs):
    model.train()
    for X, y in training_dataloader:
        if args.model != "Lin":
            X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        alpha, grads, ps, iters, ts = optimizer.step()
        scheduler.momentum_step(optimizer=optimizer, epoch=epoch)
        
        prox_grad_norm = 0.0
        for grad, p in zip(grads, ps):
            if grad != None and p != None:
                if p.ndim == 3:
                    dim = (0,2)
                elif p.ndim == 2:
                    dim = (0)

                p_ = p.sub(grad)
                prox_glasso(p=p_, eta=None, alpha=alpha, lambda_=args.lambda_, dim=dim)
                p_ = p.sub(p_)
                prox_grad_norm += p_.square().sum().item()
            
        prox_grad_norm = prox_grad_norm**0.5
        iteration = sum(iters)/len(iters)
        time = sum(ts)
        
        prox_grad_norms.append(prox_grad_norm)
        iterations.append(iteration)
        times.append(time)

    model.eval()
    for param_group in optimizer.param_groups:
        lr = param_group['lr'] 
        if args.optimizer == "RAMDA":
            momentum = param_group['momentum']
        elif args.optimizer == "ProxAdamW":
            momentum = param_group['betas']
            
    training_objective = 0.0
    training_accuracy = 0.0
    with torch.no_grad():
        for X, y in training_dataloader:
            if args.model != "Lin":
                X, y = X.cuda(), y.cuda()
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
            if dim == (0,2,3):
                dim = (0,2)
                reg_scaling = (p.numel()/p.shape[1])**0.5
                training_objective += reg_scaling*args.lambda_*(torch.linalg.norm(p.view(p.shape[0], p.shape[1], -1), dim=dim).sum().item())
            if dim == (0):
                reg_scaling = (p.numel()/p.shape[1])**0.5
                training_objective += reg_scaling*args.lambda_*(torch.linalg.norm(p, dim=dim).sum().item()) 
                
    validation_accuracy = 0.0
    with torch.no_grad():
        for X, y in testing_dataloader:
            if args.model != "Lin":
                X, y = X.cuda(), y.cuda()
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
        
    # (channel-wise + input-wise) structured sparsity  
    num_nonsparse_groups = 0.0
    num_groups = 0.0
    for group in optimizer_grouped_parameters:
        dim = group["dim"]
        for p in group["params"]:
            if dim == (0,2,3):
                num_nonsparse_groups += p.count_nonzero(dim=dim).count_nonzero().item()
                num_groups += p.shape[1]
            elif dim == (0):
                num_nonsparse_groups += p.count_nonzero(dim=dim).count_nonzero().item()
                num_groups += p.shape[1]
    structured_sparsity = 1.0-(num_nonsparse_groups/num_groups)
    
    # weighted (channel-wise + input-wise) structured sparsity
    weighted_num_nonsparse_groups = 0.0
    weighted_num_groups = 0.0
    for group in optimizer_grouped_parameters:
        dim = group["dim"]
        for p in group["params"]:
            if dim == (0,2,3):
                group_size = p.shape[0]*p.shape[2]*p.shape[3]
                weighted_num_nonsparse_groups += p.count_nonzero(dim=dim).count_nonzero().item()*group_size
                weighted_num_groups += p.shape[1]*group_size
            elif dim == (0):
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
    print("subproblem prox grad norm: {}".format(prox_grad_norm)) 
    print("subproblem iteration: {}".format(iteration))
    print("subproblem time: {}".format(time))

    if args.early_stopping:
        f = open(args.path+'Results/Presentation/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_EarlyStopping_presentation_'+str(args.seed)+'.txt', 'w+')  
    else:
        f = open(args.path+'Results/Presentation/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_NoEarlyStopping_presentation_'+str(args.seed)+'.txt', 'w+')          

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
    f.write("max iters: {}".format(args.max_iters)+'\n')
    f.write("early stopping: {}".format(args.early_stopping)+'\n')
    f.write("milestones: {}".format(args.milestones)+'\n')
    f.write("gamma: {}".format(args.gamma)+'\n')
    f.write("path: {}".format(args.path)+'\n')
    f.write("seed: {}".format(args.seed)+'\n')

    if args.optimizer == "RAMDA":
        for i, r in enumerate(zip(lrs, momentums, training_objectives, training_accuracies, validation_accuracies, unstructured_sparsities, structured_sparsities, weighted_structured_sparsities)):
            f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f}\ttraining objective:{:<20.15f}\ttraining accuracy:{:<20.15f}\tvalidation accuracy:{:<20.15f}\tunstructured sparsity:{:<20.15f}\tstructured sparsity:{:<20.15f}\tweighted structured sparsity:{:<20.15f}".format((i+1), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])+'\n')

    elif args.optimizer == "ProxAdamW":
        for i, r in enumerate(zip(lrs, momentums, training_objectives, training_accuracies, validation_accuracies, unstructured_sparsities, structured_sparsities, weighted_structured_sparsities)):
            f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f},{:<20.15f}\ttraining objective:{:<20.15f}\ttraining accuracy:{:<20.15f}\tvalidation accuracy:{:<20.15f}\tunstructured sparsity:{:<20.15f}\tstructured sparsity:{:<20.15f}\tweighted structured sparsity:{:<20.15f}".format((i+1), r[0], r[1][0], r[1][1], r[2], r[3], r[4], r[5], r[6], r[7])+'\n')    

    f.close()
    
    if args.early_stopping:
        f = open(args.path+'Results/ForPlotting/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_EarlyStopping_forplotting_'+str(args.seed)+'.txt', 'w+')
    else:
        f = open(args.path+'Results/ForPlotting/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_NoEarlyStopping_forplotting_'+str(args.seed)+'.txt', 'w+')

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
    
    if args.early_stopping:
        torch.save(model.state_dict(), args.path+'Saved_Models/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_EarlyStopping_'+str(args.seed)+'.pt')
    else:
        torch.save(model.state_dict(), args.path+'Saved_Models/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_NoEarlyStopping_'+str(args.seed)+'.pt')        
    
    # schedule learning rate and (restart for RAMDA)
    scheduler.step(optimizer=optimizer, epoch=epoch)

if args.early_stopping:
    f = open(args.path+'Results/Subproblem_Presentation/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_EarlyStopping_presentation_'+str(args.seed)+'.txt', 'w+')  
else:
    f = open(args.path+'Results/Subproblem_Presentation/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_NoEarlyStopping_presentation_'+str(args.seed)+'.txt', 'w+')  
    
f.write("final prox grad norm: {}".format(prox_grad_norm)+'\n')
f.write("final iteration: {}".format(iteration)+'\n')
f.write("final time: {}".format(time)+'\n')
    
for i, r in enumerate(zip(prox_grad_norms, iterations, times)):
    f.write("subproblem iteration:{:<10d}\tprox grad norm:{:<20.15f}\titeration:{:<5}\ttime:{:<20.15f}".format((i+1), r[0], r[1], r[2])+'\n')
    
f.close()

if args.early_stopping:
    f = open(args.path+'Results/Subproblem_ForPlotting/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_EarlyStopping_forplotting_'+str(args.seed)+'.txt', 'w+')
else:
    f = open(args.path+'Results/Subproblem_ForPlotting/'+args.optimizer+'_'+str(args.max_iters)+'_'+args.model+'_on_'+args.dataset+'_NoEarlyStopping_forplotting_'+str(args.seed)+'.txt', 'w+')
    
f.write('subproblem prox grad norm\n')
for i, grad_norm in enumerate(prox_grad_norms):
    f.write("iteration {}: {}".format((i+1), prox_grad_norm)+'\n')

f.write('subproblem iteration\n')
for i, iteration in enumerate(iterations):
    f.write("iteration {}: {}".format((i+1), iteration)+'\n')

f.write('subproblem time\n')
for i, time in enumerate(times):
    f.write("iteration {}: {}".format((i+1), time)+'\n')  
    
f.close()
