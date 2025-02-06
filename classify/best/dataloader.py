"""
作者：ztf08
名称：dataloader.py
说明：
日期：2022/5/24 16:45
"""
import sys
from random import random
import glob
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset

sys.dont_write_bytecode = True


class SampleParas:
    def __init__(self):
        self.patchdims = []
        self.patchdims = []
        self.patchstrides = []

        self.posselnum = 10
        self.negselnum = 5
        self.samplenum = 10

        self.organname = None
        self.stage = None

        self.imgdim = 2
        self.nclass = 2
        self.meanvalue = None
        self.negids = []
        self.sortids = []


sampleparas = SampleParas()
CH = 128  # 128
sampleparas.patchdims = [48, CH, 224]  # [32,128,128]
sampleparas.samplenum = 1  # 每个样本上只采样一个patch
sampleparas.imgdim = 3
sampleparas.nclass = 2
sampleparas.negids = []


def sample_random_patch_3D_witergt(img, patchdims, samplenum, mask=None, igmask=None):
    slice_, height_, width_ = img.shape
    patches = np.empty(shape=(0, 1, patchdims[0], patchdims[1], patchdims[2]), dtype=np.float32)
    # [0,1,64,128,128]
    patchcoors = np.zeros((1, 6), dtype=np.uint16)

    nostart = 0
    counter = 0
    ccot = 1
    index = np.argwhere(mask > 0)

    maxv = np.zeros((3, 1))
    minv = np.zeros((3, 1))  # 存放index每列最大最小值
    if index.shape[0] > 0:
        for ii in range(0, 3):
            maxv[ii] = np.max(index[:, ii])
            minv[ii] = np.min(index[:, ii])

    while counter < samplenum:  ##随机采样1个样本
        if ccot > 1000:
            print("can't locate mask")
            break

        z_min = np.random.randint(max(minv[0][0]-int(patchdims[0] / 2)+10,int(patchdims[0] / 2)),
                                  min(maxv[0][0]+int(patchdims[0] / 2)-10,slice_ - int(patchdims[0] / 2)))
        y_min = np.random.randint(max(minv[1][0]-int(patchdims[1] / 2)+10,int(patchdims[1] / 2)),
                                  min(maxv[1][0]+int(patchdims[1] / 2)-10,height_ - int(patchdims[1] / 2)))
        x_min = np.random.randint(max(minv[2][0]-int(patchdims[2] / 2)+10,int(patchdims[2] / 2)),
                                  min(maxv[2][0]+int(patchdims[2] / 2)-10,width_ - int(patchdims[2] / 2)))

        z0 = z_min - int(patchdims[0] / 2)
        z1 = z_min + int(patchdims[0] / 2)
        y0 = y_min - int(patchdims[1] / 2)
        y1 = y_min + int(patchdims[1] / 2)
        x0 = x_min - int(patchdims[2] / 2)
        x1 = x_min + int(patchdims[2] / 2)

        if mask is not None:
            curpatchmask = mask[z0:z1,y0:y1,x0:x1]
            if np.count_nonzero(curpatchmask) < 1000:  ##如果没有label,则当前patch 不选。
                ccot = ccot + 1
                continue

        counter += 1

        curpatch = img[z0:z1, y0:y1, x0:x1].reshape(
            [1, 1, patchdims[0], patchdims[1], patchdims[2]])  # 进行数据的reshape 操作。
        patches = np.append(patches, curpatch, axis=0)
        if nostart == 0:
            patchcoors[0, :] = [z0, z1, y0, y1, x0, x1]
            nostart = 1
        else:
            patchcoors = np.row_stack((patchcoors, [z0, z1, y0, y1, x0, x1]))  ##直接是stack 成数组。


    patches = patches.astype(np.float32)
    return patches, patchcoors


def sample_label_with_coor_3D(label, patchdims, patchcoors, nclass=2):
    samplenum = patchcoors.shape[0]
    if nclass == 2:
        patches = np.empty(shape=(0, 1, patchdims[0], patchdims[1], patchdims[2]), dtype=np.uint8)  # 如果是两分类的情况
    else:
        patches = np.empty(shape=(0, nclass, patchdims[0], patchdims[1], patchdims[2]), dtype=np.uint8)  # 如果是多分类的情况
    for i in range(samplenum):
        if nclass == 2:
            currpatches = label[patchcoors[i, 0]:patchcoors[i, 1], patchcoors[i, 2]:patchcoors[i, 3],
                          patchcoors[i, 4]:patchcoors[i, 5]].reshape([1, 1, patchdims[0], patchdims[1], patchdims[2]])
        else:
            currpatches = label[patchcoors[i, 0]:patchcoors[i, 1], patchcoors[i, 2]:patchcoors[i, 3],
                          patchcoors[i, 4]:patchcoors[i, 5]]
            currpatches = dim_2_categorical(currpatches, nclass)
            currpatches = currpatches.reshape([1, nclass, patchdims[0], patchdims[1], patchdims[2]])
        patches = np.append(patches, currpatches, axis=0)
    patches = patches.astype(np.uint8)
    return patches


def dim_2_categorical(label, numclass):
    dims = label.ndim
    # print(label.shape)
    if dims == 2:
        height_, width_ = label.shape
        exlabel = np.zeros((numclass, height_, width_))
        for i in range(0, numclass):
            exlabel[i,] = np.asarray(label == i).astype(np.uint8)
    elif dims == 3:  ##如果是三类，则生成3个tensors
        slice_, height_, width_ = label.shape
        exlabel = np.zeros((numclass, slice_, height_, width_))
        for i in range(0, numclass):
            exlabel[i,] = np.asarray(label == i).astype(np.uint8)

    return exlabel


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

    def __init__(self, data_list, preimgpath, labelpath):
        self.data_list = data_list
        self.preimgpath = preimgpath
        self.labelpath = labelpath

    def __getitem__(self, item):
        preimgfile = self.data_list['ID'][item]

        midname = preimgfile
        # remove the line break character
        if midname.count(".") > 0:
            midname = midname[0:midname.index('.')]
        else:
            midname = midname.strip('\n')
        # data_tensor=torch.ones([1,48,128,128])
        # mask_tensor=torch.ones([48,128,128])

        if 1:
        # if midname=='LIU BI XIAN':
            preimgraw = sitk.ReadImage(self.preimgpath + "/" + midname + ".nii.gz")
            preimg = sitk.GetArrayFromImage(preimgraw)

            labelraw = sitk.ReadImage(self.labelpath + "/" + midname + ".nii.gz")
            label = sitk.GetArrayFromImage(labelraw)

            if preimg.shape != label.shape:  ##如果shape 不一样，则不处理。
                print('orgimg shape not matching')

            normpreimg = (preimg - np.mean(preimg)) / np.std(preimg)
            try :
                patches, patchcoors = sample_random_patch_3D_witergt(normpreimg, sampleparas.patchdims, sampleparas.samplenum,
                                                                     mask=label, igmask=label)  # 通过itergt进行了采样##

                patchlabeles = sample_label_with_coor_3D(label, sampleparas.patchdims, patchcoors)  ##随机采样出label 块。

                patches = np.squeeze(patches)
                patchlabeles = np.squeeze(patchlabeles)
                # print(patchlabeles.shape)
                imgs_train = np.empty(shape=(0, sampleparas.patchdims[0], sampleparas.patchdims[1], sampleparas.patchdims[2]),
                                      dtype=np.float32)
                # print(imgs_train.shape)
                # #print(imgs_train.shape)
                imgs_label_train = np.empty(
                    shape=(0, sampleparas.patchdims[0], sampleparas.patchdims[1], sampleparas.patchdims[2]),
                    dtype=np.uint8)

                imgs_train = np.append(imgs_train, np.expand_dims(patches, axis=0), axis=0)
                # print(imgs_train.shape)
                imgs_label_train = np.append(imgs_label_train, np.expand_dims(patchlabeles, axis=0), axis=0)
                imgs_label_train = np.squeeze(imgs_label_train)

                data = imgs_train  # data[np.newaxis, :, :, :] #增加新的维度
                mask = imgs_label_train

                mask_tensor = torch.from_numpy(mask).long()  ### 转换为torch格式；
                data_tensor = torch.from_numpy(data)  ###
                self.pdata = data_tensor  ##
                self.pmask = mask_tensor
                return midname,data_tensor, mask_tensor  # 遍历一次，拿到的数据是怎么样的？
            except:
                print('error:',midname)
                # return data_tensor, mask_tensor
        # else:
        #     return data_tensor, mask_tensor



    @staticmethod
    def normalize(data):
        data = data.astype(np.float32)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def __len__(self):
        return len(self.data_list)

class SegSet(Dataset):

    def __init__(self, data_list, preimgpath, labelpath):
        self.data_list = data_list
        self.preimgpath = preimgpath
        self.labelpath = labelpath

    def __getitem__(self, item):
        fname = self.data_list['id'][item]

        preimgraw = sitk.ReadImage(self.preimgpath + "/" + fname)
        preimg = sitk.GetArrayFromImage(preimgraw)
        labelraw = sitk.ReadImage(self.labelpath + "/" + fname)
        label_seg = sitk.GetArrayFromImage(labelraw).astype(float)
        # preimg = (preimg - np.mean(preimg)) / np.std(preimg)
        data_tensor = torch.from_numpy(preimg).float()
        data_tensor = torch.unsqueeze(data_tensor,dim=0)

        return fname,data_tensor, label_seg # 遍历一次，拿到的数据是怎么样的？

    @staticmethod
    def normalize(data):
        data = data.astype(np.float32)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def __len__(self):
        return len(self.data_list)




