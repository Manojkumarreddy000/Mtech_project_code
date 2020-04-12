import torch
import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision.utils import make_grid
from torchvision import transforms

import matplotlib.pyplot as plt

def default_list_reader(fileList):
    imgPath = []
    labels = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            path, label = line.strip().split(' ')
            imgPath.append(path)              
            labels.append(int(label))

    return imgPath, labels

class VGG10Dataset(Dataset):
    def __init__(self, root, fileList, train=True, transform=None, list_reader=default_list_reader):
        self.root = root
        self.train = train
        self.transform = transform

        if train:
            self.train_paths, self.train_labels = list_reader(fileList)
            self.train_paths = self.train_paths
            self.train_labels = self.train_labels
            self.nimgs = len(self.train_paths)
        else:
            self.val_paths, self.val_labels = list_reader(fileList)
            self.nimgs = len(self.val_paths)

    def __getitem__(self, index):
        if self.train:
            imgPath, label = self.train_paths[index], self.train_labels[index]
        else:
            imgPath, label = self.val_paths[index], self.val_labels[index]

        img = Image.open(os.path.join(self.root, imgPath))           
            
        if self.transform is not None:
            img = self.transform(img)

        img = torch.from_numpy((np.array(img) - 127.5) / 128.0).permute(2, 0, 1).float()
            
        target = torch.LongTensor([label])
                        
        return img, target
        
    def __len__(self):        
        return self.nimgs

class SiameseVGG10(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    def __init__(self, vgg10_dataset): 
        self.vgg10_dataset = vgg10_dataset
        self.root = vgg10_dataset.root
        self.train = self.vgg10_dataset.train
        self.transform = self.vgg10_dataset.transform

        if self.train:
            self.train_labels = self.vgg10_dataset.train_labels
            self.train_paths = self.vgg10_dataset.train_paths
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.val_labels = self.vgg10_dataset.val_labels
            self.val_paths = self.vgg10_dataset.val_paths
            self.labels_set = set(self.val_labels)
            self.label_to_indices = {label: np.where(np.array(self.val_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.val_labels[i]]),
                               1]
                              for i in range(0, len(self.val_paths), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.val_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.val_paths), 2)]
            self.val_pairs = positive_pairs + negative_pairs
    
    def __getitem__(self, index):

        if self.train:
            target = np.random.randint(0, 2)
            img1_path, label1 = self.train_paths[index], self.train_labels[index]

            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2_path = self.train_paths[siamese_index]

        else:
            img1_path = self.val_paths[self.val_pairs[index][0]]
            img2_path = self.val_paths[self.val_pairs[index][1]]
            target = self.val_pairs[index][2]

        img1 = Image.open(os.path.join(self.root, img1_path))
        img2 = Image.open(os.path.join(self.root, img2_path))        

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        img1 = torch.from_numpy((np.array(img1) - 127.5) / 128.0).permute(2, 0, 1)
        img2 = torch.from_numpy((np.array(img2) - 127.5) / 128.0).permute(2, 0, 1)
        
        return (img1, img2), target

    def __len__(self):
        return len(self.vgg10_dataset)


class TripletVGG10(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, vgg10_dataset):
        self.vgg10_dataset = vgg10_dataset
        self.root = vgg10_dataset.root
        self.train = self.vgg10_dataset.train
        self.transform = self.vgg10_dataset.transform

        if self.train:
            self.train_labels = self.vgg10_dataset.train_labels
            self.train_paths = self.vgg10_dataset.train_paths
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0]
                                     for label in self.labels_set}

        else:
            self.val_labels = self.vgg10_dataset.val_labels
            self.val_paths = self.vgg10_dataset.val_paths
            # generate fixed triplets for testing
            self.labels_set = set(self.val_labels)
            self.label_to_indices = {label: np.where(np.array(self.val_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.val_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.val_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.val_paths))]
            self.val_triplets = triplets

    def __getitem__(self, index):

        if self.train:
            img1_path, label1 = self.train_paths[index], self.train_labels[index]
            positive_index = index

            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])

            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_path = self.train_paths[positive_index]
            img3_path = self.train_paths[negative_index]

        else:
            img1_path = self.val_paths[self.val_triplets[index][0]]
            img2_path = self.val_paths[self.val_triplets[index][1]]
            img3_path = self.val_paths[self.val_triplets[index][2]]

        img1 = Image.open(os.path.join(self.root, img1_path))
        img2 = Image.open(os.path.join(self.root, img2_path))
        img3 = Image.open(os.path.join(self.root, img3_path))
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)  

        img1 = torch.from_numpy((np.array(img1) - 127.5) / 128.0).permute(2, 0, 1)
        img2 = torch.from_numpy((np.array(img2) - 127.5) / 128.0).permute(2, 0, 1)     
        img3 = torch.from_numpy((np.array(img3) - 127.5) / 128.0).permute(2, 0, 1)     

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.vgg10_dataset)

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(np.array(self.labels)))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

if __name__=='__main__':
    #breakpoint()

    v = VGG10Dataset('data', 'vgg10trainlist.txt')
    vs = SiameseVGG10(v)
    vt = TripletVGG10(v)

    vsi = iter(vs)
    vti = iter(vt)

    v1 = VGG10Dataset('data', 'vgg10vallist.txt', train=False)
    v1s = SiameseVGG10(v1)
    v1t = TripletVGG10(v1)

    v1si = iter(v1s)
    v1ti = iter(v1t)

    img_list = []
    target_list = []
    for i, (data, target) in enumerate(v1si):        
        if i>=16:
            break
        im1, im2 = data[0], data[1]
        im1, im2 = (im1*128+127.5).to(dtype=torch.uint8), (im2*128+127.5).to(dtype=torch.uint8)
        img_list.extend([im1, im2])
        target_list.append(target) 
    
    print(target_list)
    
    """for i, (data, _) in enumerate(v1ti):
        if i>=16:
            break
        im1, im2, im3 = data[0], data[1], data[2]
        im1, im2, im3 = (im1*128+127.5).to(dtype=torch.uint8), (im2*128+127.5).to(dtype=torch.uint8), (im3*128+127.5).to(dtype=torch.uint8)
        img_list.extend([im1, im2, im3])"""
        

    img_grid = make_grid(img_list, nrow=4, padding=10)
    #img_grid = make_grid(img_list, nrow=6, padding=10)
    show(img_grid)
    plt.show()








