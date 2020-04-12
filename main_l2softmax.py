#Sairam
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler

from datasets import VGG10Dataset
from networks import EmbeddingNet, ClassificationNet
from trainer_l2softmax import fit
from metrics import AccumulatedAccuracyMetric

#breakpoint()
cuda = torch.cuda.is_available()
mining_tech = 'Triplet' #Siamese, Triplet, Doppleganger
batch_size = 81
margin = 1.
lr = 0.01
n_epochs = 20
log_interval = 50
nclasses = 854
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(hue=0.1),
                                     transforms.RandomGrayscale(p=0.1)]) # we dont add ToTensor as it is done using from_numpy in the dataset class

dataset_tr = VGG10Dataset('data', 'vgg10trainlist.txt', transform=train_transform)
dataset_val = VGG10Dataset('data', 'vgg10vallist.txt', train=False)

loss_fn = nn.CrossEntropyLoss()

train_loader = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, **kwargs)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **kwargs)

embedding_net = EmbeddingNet()
model = ClassificationNet(embedding_net, nclasses)
if cuda:
    model = nn.DataParallel(model).cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-5, last_epoch=-1)

fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()], mining_tech='RandomSampling_l2_parametrized')

