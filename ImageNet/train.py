# horovod
# overview: 
# https://horovod.readthedocs.io/en/stable/pytorch.html
# smaple code: 
# https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py
# https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
# https://github.com/pytorch/examples/tree/main/imagenet

import torch
import torchvision
import argparse
import horovod.torch as hvd
import os
import sys 
sys.path.append("..")

from Core.group import group_model
from Core.optimizer import ProxSGD, ProxAdamW, RMDA, RAMDA
from Core.scheduler import multistep_param_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='./Data')
parser.add_argument('--batch-size', type=int, default=32) # batch size per worker
parser.add_argument('--num-workers', type=int, default=32)
parser.add_argument('--smoothing', type=float, default=1e-1)
parser.add_argument('--optimizer', type=str, default='MSGD')
parser.add_argument('--fp16-allreduce', action='store_true', default=False)
parser.add_argument('--use-mixed-precision', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--momentum', type=float, default=1e-2)
parser.add_argument('--lambda_', type=float, default=0.0)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--milestones', type=int, nargs='+', default=[i for i in range(40, 160, 40)])
parser.add_argument('--gamma', type=float, default=1e-1)
parser.add_argument('--path', type=str, default='./')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

hvd.init() # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py#L153
torch.cuda.set_device(hvd.local_rank()) # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py#L158
torch.set_num_threads(args.num_workers) # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py#L208

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.benchmark = True # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2

assert args.lambda_ == 0.0 or args.weight_decay == 0.0

# https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
# https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py#L210-L246
# https://github.com/pytorch/examples/blob/main/imagenet/main.py#L231-L252
# https://github.com/IST-DASLab/ACDC/blob/main/utils/datasets.py#L162-L192

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
training_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(),
                                                      normalize])
testing_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                     torchvision.transforms.CenterCrop(224),
                                                     torchvision.transforms.ToTensor(),
                                                     normalize])

training_dataset = torchvision.datasets.ImageFolder(root=args.data_root+'/train/', 
                                                    transform=training_transforms)
training_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, 
                                                                   num_replicas=hvd.size(), 
                                                                   rank=hvd.rank())
training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                                  batch_size=args.batch_size, 
                                                  sampler=training_sampler,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)

testing_dataset = torchvision.datasets.ImageFolder(root=args.data_root+'/val/', 
                                                   transform=testing_transforms)
testing_sampler = torch.utils.data.distributed.DistributedSampler(testing_dataset, 
                                                                  num_replicas=hvd.size(), 
                                                                  rank=hvd.rank())
testing_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset, 
                                                 batch_size=args.batch_size, 
                                                 sampler=testing_sampler,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

# https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
model = torchvision.models.resnet50()
model.load_state_dict(torch.load(args.path+'Models/ResNet50_{}.pt'.format(args.seed))) # the same initialization
model.cuda()


# Label Smoothing is proposed by https://arxiv.org/abs/1512.00567
# some explanations: 
# https://arxiv.org/abs/1906.02629
# When Does Label Smoothing Help?
# Rafael Müller, Simon Kornblith, Geoffrey Hinton

# https://github.com/VITA-Group/GraNet/blob/main/ImageNet/smoothing.py
# https://github.com/IST-DASLab/ACDC/blob/main/utils/jsd_loss.py#L7-L27
class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

criterion = LabelSmoothing(smoothing=args.smoothing)

assert args.weight_decay == 0.0 or args.lambda_ == 0.0

optimizer_grouped_parameters = group_model(model=model, name="ResNet", lambda_=args.lambda_)
if args.optimizer == "MSGD":
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=9e-1)
elif args.optimizer == "ProxSGD" or args.optimizer == "ProxAdamW" or args.optimizer == "RMDA" or args.optimizer == "RAMDA":
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
                          lr=args.lr,
                          momentum=args.momentum) 
scheduler = multistep_param_scheduler(name=args.optimizer, optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)        
# https://horovod.readthedocs.io/en/stable/api.html#horovod.torch.Compression
# https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py#L270
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# https://horovod.readthedocs.io/en/stable/api.html#horovod.torch.DistributedOptimizer
# https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py#L273

# argumnets: groups = num_groups https://github.com/horovod/horovod/blob/master/horovod/torch/optimizer.py#L589
# see Section 3.2 of the paper https://www.usenix.org/conference/nsdi22/presentation/romero
optimizer = hvd.DistributedOptimizer(optimizer,
                                     compression=compression,
                                     op=hvd.Average)

# https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py#L288
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

if args.use_mixed_precision:
    scaler = torch.cuda.amp.GradScaler() # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py#L237

lrs = []
momentums = []
training_losses = []
training_accuracies = []
validation_accuracies = []
unstructured_sparsities = []
structured_sparsities = [] 
weighted_structured_sparsities = []

for epoch in range(args.epochs):
    model.train()
    training_sampler.set_epoch(epoch) # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py#L74            
    training_accuracy = 0.0
    training_loss = 0.0
    for X, y in training_dataloader:
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        if args.use_mixed_precision:
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py#L79  
                output = model(X)
                loss = criterion(output, y)
            # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py#L83-L91
            scaler.scale(loss).backward() 
            optimizer.synchronize() 
            scaler.unscale_(optimizer)
            with optimizer.skip_synchronize():
                scaler.step(optimizer)
            scaler.update()
            # schedule momentum for RMDA and RAMDA 
            scheduler.momentum_step(optimizer=optimizer, epoch=epoch)
            training_loss += loss.item()*len(y)
            y_hat = output.argmax(dim=1)
            training_accuracy += y_hat.eq(y.view_as(y_hat)).float().sum().item()
        else:
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
     
    # allreduce to calculate training loss and training accuracy 
    training_loss = (hvd.allreduce(torch.tensor([training_loss]).cuda(), op=hvd.Sum)/len(training_dataset)).item()
    training_accuracy = (hvd.allreduce(torch.tensor([training_accuracy]).cuda(), op=hvd.Sum)/len(training_dataset)).item()
        
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
        # allreduce to calculate validation accuracy
        validation_accuracy = (hvd.allreduce(torch.tensor([validation_accuracy]).cuda(), op=hvd.Sum)/len(testing_dataset)).item()

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
    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)
    validation_accuracies.append(validation_accuracy)
    unstructured_sparsities.append(unstructured_sparsity)
    structured_sparsities.append(structured_sparsity)
    weighted_structured_sparsities.append(weighted_structured_sparsity)
        
    if hvd.rank() == 0: # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py#L148       
        print("optimizer: {}".format(args.optimizer))
        print("epochs: {}".format(epoch+1))
        print("learning rate: {}".format(lr))
        print("momentum: {}".format(momentum))
        print("lambda_: {}".format(args.lambda_))
        print("weight decay: {}".format(args.weight_decay))
        print("training loss: {}".format(training_loss))
        print("training accuracy: {}".format(training_accuracy))
        print("validation accuracy: {}".format(validation_accuracy))
        print("unstructured sparsity: {}".format(unstructured_sparsity))
        print("structured sparsity: {}".format(structured_sparsity))
        print("weighted structured sparsity: {}".format(weighted_structured_sparsity))
        
        f = open(args.path+'Results/Presentation/'+args.optimizer+'_ResNet50_on_ImageNet_presentation_'+str(args.seed)+'.txt', 'w+')  

        f.write("final training loss: {}".format(training_loss)+'\n')
        f.write("final training accuracy: {}".format(training_accuracy)+'\n')
        f.write("final validation accuracy: {}".format(validation_accuracy)+'\n')
        f.write("final unstructured sparsity: {}".format(unstructured_sparsity)+'\n')
        f.write("final structured sparsity: {}".format(structured_sparsity)+'\n')
        f.write("final weighted structured sparsity: {}".format(weighted_structured_sparsity)+'\n')
        
        f.write("data root: {}".format(args.data_root)+'\n')
        f.write("batch size: {}".format(args.batch_size)+'\n')
        f.write("num workers: {}".format(args.num_workers)+'\n')
        f.write("smoothing: {}".format(args.smoothing)+'\n')
        f.write("optimizer: {}".format(args.optimizer)+'\n')
        f.write("fp16 allreduce: {}".format(args.fp16_allreduce)+'\n')
        f.write("use mixed precision: {}".format(args.use_mixed_precision)+'\n')
        f.write("epochs: {}".format(args.epochs)+'\n')
        f.write("lr: {}".format(args.lr)+'\n')
        f.write("lambda_: {}".format(args.lambda_)+'\n')
        f.write("weight decay: {}".format(args.weight_decay)+'\n')
        f.write("milestones: {}".format(args.milestones)+'\n')
        f.write("gamma: {}".format(args.gamma)+'\n')
        f.write("path: {}".format(args.path)+'\n')
        f.write("seed: {}".format(args.seed)+'\n')

        if args.optimizer == "MSGD" or args.optimizer == "ProxSGD" or args.optimizer == "RMDA" or args.optimizer == "RAMDA":
            for i, r in enumerate(zip(lrs, momentums, training_losses, training_accuracies, validation_accuracies, unstructured_sparsities, structured_sparsities, weighted_structured_sparsities)):
                f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f}\ttraining loss:{:<20.15f}\ttraining accuracy:{:<20.15f}\tvalidation accuracy:{:<20.15f}\tunstructured sparsity:{:<20.15f}\tstructured sparsity:{:<20.15f}\tweighted structured sparsity:{:<20.15f}".format((i+1), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])+'\n')
        elif args.optimizer == "ProxAdamW":
            for i, r in enumerate(zip(lrs, momentums, training_losses, training_accuracies, validation_accuracies, unstructured_sparsities, structured_sparsities, weighted_structured_sparsities)):
                f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f},{:<20.15f}\ttraining loss:{:<20.15f}\ttraining accuracy:{:<20.15f}\tvalidation accuracy:{:<20.15f}\tunstructured sparsity:{:<20.15f}\tstructured sparsity:{:<20.15f}\tweighted structured sparsity:{:<20.15f}".format((i+1), r[0], r[1][0], r[1][1], r[2], r[3], r[4], r[5], r[6], r[7])+'\n')    

        f.close()
    
        f = open(args.path+'Results/ForPlotting/'+args.optimizer+'_ResNet50_on_ImageNet_forplotting_'+str(args.seed)+'.txt', 'w+')

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
        
        torch.save(model.state_dict(), args.path+'Saved_Models/'+args.optimizer+'_ResNet50_on_ImageNet_'+str(args.seed)+'.pt')
     
    # schedule learning rate and (restart for RMDA and RAMDA)
    scheduler.step(optimizer=optimizer, epoch=epoch)