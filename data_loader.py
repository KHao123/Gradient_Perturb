"""
   CIFAR-10 CIFAR-100, Tiny-ImageNet data loader
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCECIFAR100
from data.ClassAwareSampler import get_sampler
# from data.ReverseSampler import ImbalancedDatasetSampler,callback_get_label
from data.skinDatasetFolder import skinDatasetFolder
from data.skin198datasets import SD198

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    img_num = torch.zeros([0])

    if params.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='/data/kanghao/datasets/data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data/kanghao/datasets/data-cifar10', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='/data/kanghao/datasets/data-cifar100', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='/data/kanghao/datasets/data-cifar100', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset == 'tiny_imagenet':
        data_dir = './data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/images/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    elif params.dataset == 'imbalance_cifar100':
        trainset = IMBALANCECIFAR100(types, imbalance_ratio=params.cifar_imb_ratio, root='/data/kanghao/datasets/data-cifar100')
        img_num = trainset.img_num
        img_num = torch.tensor(img_num,dtype=torch.float)

        devset = IMBALANCECIFAR100(types, imbalance_ratio=params.cifar_imb_ratio, root='/data/kanghao/datasets/data-cifar100')

    elif params.dataset == 'imbalance_cifar10':
        trainset = IMBALANCECIFAR10(types, imbalance_ratio=params.cifar_imb_ratio, root='/data/kanghao/datasets/data-cifar10')
        img_num = trainset.img_num
        img_num = torch.tensor(img_num,dtype=torch.float)
        
        devset = IMBALANCECIFAR10(types, imbalance_ratio=params.cifar_imb_ratio, root='/data/kanghao/datasets/data-cifar10')
    elif params.dataset == 'skin7':
        trainset = skinDatasetFolder(train=True, iterNo=params.iterNo, data_dir='/data/Public/Datasets/Skin7')
        devset = skinDatasetFolder(train=False, iterNo=params.iterNo, data_dir='/data/Public/Datasets/Skin7')

        img_num = trainset.img_num
        img_num = torch.tensor(img_num,dtype=torch.float)
    elif params.dataset == 'sd198':
        trainset = SD198(train=True, transform=None, iter_no=params.iterNo, data_dir='/data/Public/Datasets/SD198')
        devset = SD198(train=False, transform=None, iter_no=params.iterNo, data_dir='/data/Public/Datasets/SD198')
        img_num = trainset.img_num
        img_num = torch.tensor(img_num,dtype=torch.float)
    if params.resample == "yes":
        sampler = get_sampler()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, sampler= sampler(trainset, 4),
                                                shuffle=False, num_workers=params.num_workers)

        devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                                shuffle=False, num_workers=params.num_workers)
    # elif params.resample == "reverse":
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, sampler= ImbalancedDatasetSampler(trainset,callback_get_label=callback_get_label),
    #                                             shuffle=False, num_workers=params.num_workers)
    #     devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
    #                                             shuffle=False, num_workers=params.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                                shuffle=True, num_workers=params.num_workers)

        devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                                shuffle=False, num_workers=params.num_workers)


    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl,trainset.cls_num,img_num


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if params.dataset=='cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset=='cifar100':
        trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset == 'tiny_imagenet':
        data_dir = './data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/images/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl

if __name__ == '__main__':
    json_path = os.path.join('experiments/imbalance_experiments/resample_resnet18/', 'params.json')
    import utils
    params = utils.Params(json_path)
    train_dl = fetch_dataloader('train', params)
    labels = []
    for (data, label) in train_dl:
        labels.append(label)
    labels = torch.cat(labels)
    print(labels.shape)
    print(torch.unique(labels,return_counts =True))
    # for i in range(100):
    #     print((labels==i))
