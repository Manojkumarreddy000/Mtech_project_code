import torch
import numpy as np
import os
import random

nc = 27
nr = 9
nclasses = 854
nsamples_per_class = 3

def get_batch_class_indices(dg_list):
    #dg_list is a dictionary of sets with ith set containing dopplegangers for ith class
    """if not dg_list:
        return random.sample(list(range(nclasses)), nc)"""

    dg_list1 = list(dg_list.values())    

    random_indices = random.sample(list(range(nclasses)), nr)     # samples without replacement
    dg = [dg_list1[index] for index in random_indices]
     
    dg_indices = list(set([random.choice(list(s)) for s in dg if s]))
    
    if len(dg_indices) != len(random_indices):
        lft_out_indices = set(list(np.arange(nclasses))) - set(random_indices) - set(dg_indices)
        temp_indices = random.sample(lft_out_indices, len(random_indices)-len(dg_indices))
        dg_indices = dg_indices + temp_indices
    
    dg = [dg_list1[index] for index in dg_indices]
    dg_dg_indices = list(set([random.choice(list(s)) for s in dg if s]))

    if len(dg_dg_indices) != len(dg_indices):
        lft_out_indices = set(list(range(nclasses))) - set(random_indices) - set(dg_indices) - set(dg_dg_indices)
        temp_indices = random.sample(lft_out_indices, len(dg_indices)-len(dg_dg_indices))
        dg_dg_indices = dg_dg_indices + temp_indices

    batch_class_indices = random_indices + dg_indices + dg_dg_indices

    return batch_class_indices

def get_batch(dset, dg_list):

    class_indices = get_batch_class_indices(dg_list)
    if len(class_indices) < nc:
        class_indices = random.sample(list(range(nclasses)), nc)

    if dset.train:
        labels = [list(np.where(np.isin(dset.train_labels, i))[0]) for i in class_indices]
        sample_indices = [random.sample(l, nsamples_per_class) for l in labels]        
    else:
        labels = [list(np.where(np.isin(dset.val_labels, i))[0]) for i in class_indices]
        sample_indices = [random.sample(l, nsamples_per_class) for l in labels]

    batch_indices = [index for indices in sample_indices for index in indices]
    img = []
    target = []
    for i in batch_indices:
        im, tgt = dset[i]
        img.append(im)
        target.append(tgt)
    
    images = torch.stack(img, dim=0)
    targets = torch.stack(target, dim=0)
    
    return images, targets


def fit(dset, model, loss_fn, optimizer, scheduler, n_iterations, cuda, log_interval, metrics=[], mining_tech='Doppleganger'):

    dg_list = {}
    for i in range(854):
        dg_list[i] = set()

    best_iter = -1
    best_loss = 1000    

    for it in range(n_iterations):
        #print(it)
        images, targets = get_batch(dset, dg_list)
        
        # Train stage
        train_loss, metrics = train_epoch(model, images, targets, dg_list, it, loss_fn, optimizer, cuda, log_interval, metrics)
        """sum_ = 0
        for v in dg_list.values():
            sum_ += sum(v)
        print('sum is:', sum_)"""
        

        if n_iterations % 3900 == 0:
            scheduler.step()            
            if train_loss < best_loss:
                best_loss = train_loss
                best_iteration = it
                torch.save({'best_iteration': best_iteration,
                        'loss': best_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, 
                               os.path.join('checkpoints', mining_tech, 'best_model.pth'))


def train_epoch(model, images, targets, dg_list, it, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    if cuda:
        images = images.cuda()         
        targets = targets.squeeze().cuda()

    optimizer.zero_grad()
    norm_scores, embeddings = model(images)       

    _, indices = norm_scores.topk(2, dim=1)
    indices1 = [index[0].item() if index[0] != targets[i] else index[1].item() for i, index in enumerate(indices)]
    for i in range(images.size(0)):
        dg_list[targets[i].item()].add(indices1[i])  # dg_list is updated here and will be visible in fit function
        
    loss_outputs = loss_fn(norm_scores, embeddings, targets)
    loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs    
    loss.backward()
    optimizer.step()

    for metric in metrics:
        metric(norm_scores, targets, loss_outputs)

    if it % log_interval == 0:
        message = f'Train Loss at iter {it}: {loss/81}'

        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
    
    return loss, metrics
