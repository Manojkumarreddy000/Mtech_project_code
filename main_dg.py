import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
import torchvision.transforms as transforms

from datasets import VGG10Dataset
from losses import DopplegangerLoss
from networks import EmbeddingNet, ClassificationNet
from trainer_dg import fit
from metrics import AccumulatedAccuracyMetric

#breakpoint()
cuda = torch.cuda.is_available()
mining_tech = 'Doppleganger' #Siamese, Triplet, Doppleganger
batch_size = 81
margin = 1.
lr = 0.01
niterations = 78000
log_interval = 50
nclasses = 854
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(hue=0.1),
                                     transforms.RandomGrayscale(p=0.1)]) # we dont add ToTensor as it is done using from_numpy in the dataset class

dataset_tr = VGG10Dataset('data', 'vgg10trainlist.txt', transform=train_transform)
dataset_val = VGG10Dataset('data', 'vgg10vallist.txt', train=False)

loss_fn = DopplegangerLoss()

embedding_net = EmbeddingNet()
model = ClassificationNet(embedding_net, nclasses)
if cuda:
    model = nn.DataParallel(model).cuda()
    loss_fn = loss_fn.cuda()

optimizer = optim.SGD([{'params':model.parameters()}, {'params':loss_fn.parameters()}], lr=lr, nesterov=True, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-5, last_epoch=-1)

fit(dataset_tr, model, loss_fn, optimizer, scheduler, niterations, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()], mining_tech='Doppleganger')

    
    
