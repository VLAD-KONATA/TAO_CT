import math

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import SimpleITK as sitk
from torchvision import transforms


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)  # k为正数时，表示逆时针旋转90度xk，k取负数时，顺时针旋转。
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    # tr = transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=45),
    #                                         transforms.RandomVerticalFlip(p=0.5),
    #                                         transforms.RandomHorizontalFlip(p=0.5)], p=0.5)
    tr = transforms.RandomRotation(degrees=45)

    all_data = torch.cat([image, label], dim=0)
    all_data = tr(all_data)

    image = all_data[0:len(image)]
    label = all_data[len(image):].long()

    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class LAHeart(Dataset):
    """ LA Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir + '/train.list'
        test_path = self._base_dir + '/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class Pancreas(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir + '/train.list'
        test_path = self._base_dir + '/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/Pancreas_h5/" + image_name + "_norm.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool)
        image = sk_trans.resize(image, self.output_size, order=1, mode='constant', cval=0)
        label = sk_trans.resize(label, self.output_size, order=0)
        assert (np.max(label) == 1 and np.min(label) == 0)
        assert (np.unique(label).shape[0] == 2)

        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label, cls_label = sample['image'], sample['label'], sample['cls_label']
        name = sample['name']
        sample_num = 0

        maxv = np.zeros((3, 1))
        minv = np.zeros((3, 1))  # 存放index每列最大最小值
        for ii in range(0, 3):
            minv[ii] = 0
            maxv[ii] = image.shape[ii] - self.output_size[ii]

        if label.sum() != 0:
            while sample_num < 1000:
                if minv[0]==maxv[0]:
                    z0 = 0
                else:
                    z0 = int(np.random.randint(minv[0], maxv[0]))
                z1 = int(z0 + self.output_size[0])
                y0 = int(np.random.randint(minv[1], maxv[1]))
                y1 = int(y0 + self.output_size[1])
                if minv[2]==maxv[2]:
                    x0 = 0
                else:
                    x0 = int(np.random.randint(minv[2], maxv[2]))
                x1 = int(x0 + self.output_size[2])

                image0 = image[z0:z1, y0:y1, x0:x1]
                label0 = label[z0:z1, y0:y1, x0:x1]
                sample_num += 1
                if len(np.where(label0!=0)[0]) > 1000:
                    break

        else:
            z0 = int(np.random.randint(minv[0], maxv[0]) - self.output_size[0] // 2)
            z1 = int(z0 + self.output_size[0])
            y0 = int(np.random.randint(minv[1], maxv[1]) - self.output_size[1] // 2)
            y1 = int(y0 + self.output_size[1])
            x0 = int(np.random.randint(minv[2], maxv[2]) - self.output_size[2] // 2)
            x1 = int(x0 + self.output_size[2])

            image0 = image[z0:z1, y0:y1, x0:x1]
            label0 = label[z0:z1, y0:y1, x0:x1]

        return {'name': name, 'image': image0, 'label': label0, 'cls_label': cls_label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label, cls_label = sample['image'], sample['label'], sample['cls_label']
        name = sample['name']
        image, label = random_rot_flip(image, label)

        return {'name': name, 'image': image, 'label': label, 'cls_label': cls_label}


class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label, cls_label = sample['image'], sample['label'], sample['cls_label']
        name = sample['name']
        image, label = random_rotate(image, label)

        return {'name': name, 'image': image, 'label': label, 'cls_label': cls_label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma,
                        2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        name = sample['name']
        image = sample['image']
        image = image.astype(np.float32)
        label = sample['label'].astype(np.uint8)
        cls_label = sample['cls_label']
        return {'name': name, 'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                'cls_label': cls_label}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, primary_batch_size, secondary_batch_size):
        self.primary_indices = primary_indices  # 90
        self.secondary_indices = secondary_indices  # 706
        self.secondary_batch_size = secondary_batch_size  # 8
        self.primary_batch_size = primary_batch_size  # 2

        self.indice = math.ceil(len(secondary_indices) / secondary_batch_size)
        self.indice_ = len(secondary_indices) // secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_once(self.secondary_indices)
        while len(primary_iter) < len(secondary_iter):
            primary_iter = np.append(primary_iter, primary_iter)
        primary_iter = primary_iter[0:len(secondary_iter)]

        label_zuhe = grouper(primary_iter, self.primary_batch_size)
        unlabel_zuhe = grouper(secondary_iter, self.secondary_batch_size)

        all_index = []
        for (primary_batch, secondary_batch) in zip(label_zuhe, unlabel_zuhe):
            all_index.append((primary_batch + secondary_batch))
        if self.indice != self.indice_:
            add_index = tuple(primary_iter[
                              self.primary_batch_size * self.indice_:self.primary_batch_size * self.indice_ + self.primary_batch_size]) + tuple(
                secondary_iter[self.secondary_batch_size * self.indice_:])
            all_index.append(add_index)
        all_index = tuple(all_index)
        return iter(all_index)

    def __len__(self):
        return self.indice


def iterate_once(iterable):
    return np.random.permutation(iterable)  # 针对一维数据随机排列


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkBSpline):
    # sitk.sitkBSpline
    # sitk.sitkLinear
    # sitk.sitkNearestNeighbor
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()  # 原来的体素块间隔
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


class MySet(Dataset):
    """
    the dataset class receive a list that contain the data item, and each item is a dict
    with two item include data path and label path. as follow:
    data_list = [
    {
    "data": data_path_1,
    "label": label_path_1,
    ...
    }
    ]
    """

    def __init__(self, data_list, preimgpath, labelpath, transform=None):
        self.data_list = data_list
        self.preimgpath = preimgpath
        self.labelpath = labelpath
        self.transform = transform

    def __getitem__(self, item):
        preimgfile = self.data_list['id'][item]
        cls_label = self.data_list['label'][item]
        midname = preimgfile

        preimgraw = sitk.ReadImage(
            '/media/dell/SATA1/dataset/CT/origin50_1' + "/" + midname)
        preimg = sitk.GetArrayFromImage(preimgraw)
        labelraw = sitk.ReadImage('/media/dell/SATA1/dataset/CT/seg50_7.20' + "/" + midname)
        label = sitk.GetArrayFromImage(labelraw)
        cls_label = torch.tensor(cls_label).long()
        if preimg.shape[0] < 50:
            lo = 50 - preimg.shape[0]
            lo = np.zeros((lo, 150, 240))
            preimg = np.concatenate((preimg, lo), axis=0)
            label = np.concatenate((label, lo), axis=0)
        assert preimg.shape==label.shape, '大小不同'

        # preimg = (preimg - np.mean(preimg)) / np.std(preimg)

        sample = {'name': midname, 'image': preimg, 'label': label, 'cls_label': cls_label}
        if self.transform:
            sample = self.transform(sample)
        if sample['image'].shape[0] != 1:
            sample['image'] = torch.unsqueeze(sample['image'], dim=0)
            sample['label'] = torch.unsqueeze(sample['label'], dim=0)
        return sample  # 遍历一次，拿到的数据是怎么样的？

    @staticmethod
    def normalize(data):
        data = data.astype(np.float32)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def __len__(self):
        return len(self.data_list)


class TestSet(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.fnames = self.data_list['id']
        self.label = self.data_list['label']

    def __getitem__(self, item):
        preimgfile = self.fnames[item]
        midname = preimgfile

        preimgraw = sitk.ReadImage(
            '/media/dell/SATA1/dataset/CT/origin50_1' + "/" + midname)

        preimg = sitk.GetArrayFromImage(preimgraw)

        if preimg.shape[0] < 50:
            preimgraw = resize_image_itk(preimgraw, (240, 150, 50))
            preimg = sitk.GetArrayFromImage(preimgraw)

        # preimg = (preimg - np.mean(preimg)) / np.std(preimg)

        data_tensor = torch.from_numpy(preimg).float()
        data_tensor = data_tensor[0:48,0:144,0:240]
        self.pdata = data_tensor
        label = self.label[item]
        label_tensor = torch.tensor(float(label))
        return midname, data_tensor, label_tensor  # 遍历一次，拿到的数据是怎么样的？

    def __len__(self):
        return len(self.fnames)
