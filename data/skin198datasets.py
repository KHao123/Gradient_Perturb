"""datasets
"""

import sys
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

from torch.utils.data import Dataset


class SD198(Dataset):
    cls_num = 198
    def __init__(self, train=True, transform=None, iter_no=0, data_dir='/data/Public/Datasets/SD198'):       #
        
        self.train = train
        self.data_dir = os.path.join(data_dir, 'images')
        self.data, self.targets = self.get_data(iter_no, data_dir)
        self.dataset_name = 'SD198'
        class_idx_path = os.path.join(data_dir, 'class_idx.npy')
        self.classes = self.get_classes_name(class_idx_path)
        self.classes = [class_name for class_name, _ in self.classes]

        if self.train:
            self.train_labels = torch.LongTensor(self.targets)
            self.img_num = [self.targets.count(i) for i in range(self.cls_num)]
        else:
            self.test_labels = torch.LongTensor(self.targets)

        Resize_img = 300
        Crop_img = 224
        self.labels = self.targets
        # mean = [(0.592, 0.479, 0.451),
        # (0.25338554 0.08627725 0.01029419),
        # (0.10649281 0.03926077 0.06205735),
        # (0.03861747 -0.01254432 -0.04720071),
        # (0.15823184  0.02122124 -0.01373529)]

        # std = [(0.265, 0.245, 0.247),
        # (0.91431177 0.9481566  0.9131112),
        # (0.88490963 0.8685787  0.8774559),
        # (0.94373405 0.91663104 0.9304368),
        # (0.9062277 0.900465  0.9045457)
        # ]
        mean = [0.5896185, 0.4765919, 0.45172438]
        std = [0.26626918, 0.24757613, 0.24818243]

        normalized = transforms.Normalize(mean=mean,std=std)

        transform_train = transforms.Compose([
            transforms.Resize(Resize_img),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            # transforms.RandomRotation([-180, 180]),
            # transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
            transforms.RandomCrop(Crop_img),
            transforms.ToTensor(),
            normalized
        ])
        transform_test = transforms.Compose([
            transforms.Resize(Resize_img),
            transforms.CenterCrop(Crop_img),
            transforms.ToTensor(),
            normalized
        ])

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transform_train if self.train else transform_test
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.data[index]
        target = self.targets[index]
        img = pil_loader(path)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self, iter_no, data_dir):

        if self.train:
            txt = '8_2_split/train_{}.txt'.format(iter_no)
        else:
            txt = '8_2_split/val_{}.txt'.format(iter_no)

        fn = os.path.join(data_dir, txt)
        txtfile = pd.read_csv(fn, sep=" ")
        raw_data = txtfile.values

        data = []
        targets = []
        for (path, label) in raw_data:
            data.append(os.path.join(self.data_dir, path))
            targets.append(label)

        return data, targets

    def get_classes_name(self, data_dir):
        classes_name = np.load(data_dir)
        return classes_name


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == '__main__':
    mean = (0.592, 0.479, 0.451)
    std = (0.265, 0.245, 0.247)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    trainset = SD198(train=True, transform=transform, iter_no=1)

    loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8)
    for data in loader:
        images, labels = data
        print('images:', images.size())
        print('labels', labels.size())
