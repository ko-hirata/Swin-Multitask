import cv2
import numpy as np
import random
import torch

from scipy import ndimage
from torch.utils.data import Dataset
from torchvision import transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = cv2.resize(image, self.output_size)
            label = cv2.resize(label, self.output_size)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        return sample


class RandomResizedCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)

        transform = transforms.RandomResizedCrop(self.output_size[0],  interpolation=transforms.InterpolationMode.BICUBIC)
        params = transform.get_params(image, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))

        image = transforms.functional.crop(image, *params)
        image = transforms.functional.resize(image, (self.output_size[0], self.output_size[0]))

        label = transforms.functional.crop(label, *params)
        label = transforms.functional.resize(label, (self.output_size[0], self.output_size[0]))

        label = label.squeeze()
        sample = {'image': image, 'label': label.long()}
        return sample


class SimpleTransform(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = cv2.resize(image, self.output_size)
        label = cv2.resize(label, self.output_size)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class BrainctDataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.data_dir = base_dir
        self.split = split
        self.transform = transform

        self.data = np.load(self.data_dir + f'/{split}.npz')
        self.images = self.data['image']
        self.labels = self.data['label']
        self.masks = self.data['mask']
        self.masks_exist = self.data['mask_exist']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] / 255
        label = self.labels[idx]
        mask = self.masks[idx] / 255
        mask_exist = self.masks_exist[idx]

        sample = {'image': image, 'label': mask}
        if self.transform:
            sample = self.transform(sample)

        image = sample['image']
        label = label.astype(np.int32)
        mask = sample['label']

        return image, label, mask, mask_exist