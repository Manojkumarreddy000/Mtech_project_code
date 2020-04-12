#Sairam
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler

from datasets import VGG10Dataset, SiameseVGG10, TripletVGG10, BalancedBatchSampler
from networks import EmbeddingNet, SiameseNet, TripletNet
from losses import OnlineContrastiveLoss, OnlineTripletLoss
from utils import HardNegativePairSelector, RandomNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from trainer import fit

#breakpoint()
cuda = torch.cuda.is_available()
mining_tech = 'Triplet' #Siamese, Triplet, Doppleganger
batch_size = 81
margin = 1.
lr = 0.01
n_epochs = 20
log_interval = 50
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(hue=0.1),
                                     transforms.RandomGrayscale(p=0.1)]) # we dont add ToTensor as it is done using from_numpy in the dataset class

dataset_tr = VGG10Dataset('data', 'vgg10trainlist.txt', transform=train_transform)
dataset_val = VGG10Dataset('data', 'vgg10vallist.txt', train=False)


#loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
"""if mining_tech == 'Siamese':
    train_dataset = SiameseVGG10(dataset_tr)
    val_dataset = SiameseVGG10(dataset_val)
    loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
    if cuda:
        loss_fn = loss_fn.cuda()

elif mining_tech == 'Triplet':
    train_dataset = TripletVGG10(dataset_tr)
    val_dataset = TripletVGG10(datset_val)
    loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    if cuda:
        loss_fn = loss_fn.cuda()

elif mining_tech == 'Doppleganger':
    pass

else:
    raise Exception('Wrong choice for mining technique')"""

train_batch_sampler = BalancedBatchSampler(dataset_tr.train_labels, n_classes=27, n_samples=3)
val_batch_sampler = BalancedBatchSampler(dataset_val.val_labels, n_classes=27, n_samples=3)

train_loader = DataLoader(dataset_tr, batch_sampler=train_batch_sampler, **kwargs)
val_loader = DataLoader(dataset_val, batch_sampler=val_batch_sampler, **kwargs)

embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model = nn.DataParallel(model).cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-5, last_epoch=-1)

fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, mining_tech='Triplet')

