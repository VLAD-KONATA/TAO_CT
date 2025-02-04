import os

import SimpleITK as sitk
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

import ramps
from dataloader import ValSet,TestSet
from dataset import *
from resnet4 import *
from ztfloss import *

from medpy.metric import binary


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def suanscore(y_true, y_pred):
    # 自定义的f1 socre
    acc = accuracy_score(y_true, y_pred)
    recall = 0.5 * recall_score(y_true, y_pred, pos_label=0) + 0.5 * recall_score(y_true, y_pred, pos_label=1)
    f1 = 0.5 * f1_score(y_true, y_pred, pos_label=0) + 0.5 * f1_score(y_true, y_pred, pos_label=1)
    return acc, recall, f1


def Dice(seg_pre, seg_y):
    dice = ((seg_pre * seg_y).sum().item() * 2 + 1e-5) / (seg_pre.sum().item() + seg_y.sum().item() + 1e-5)
    return dice


def soft_dice(seg_pre, seg_y):
    dice = 0
    seg_y = torch.squeeze(seg_y).to(device)
    seg_pre = torch.squeeze(seg_pre).to(device)
    assert seg_pre.shape == seg_y.shape
    for i in range(1, 6):
        seg_pre_ = (seg_pre == i).int()
        seg_y_ = (seg_y == i).int()
        dice += Dice(seg_pre_, seg_y_)
    return dice / 5


def sharpening(P):
    T = 1 / 0.1
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1 * ramps.sigmoid_rampup(epoch, 40)

def cal_metric(prediction_tensor,ground_truth_tensor):
    prediction = torch.squeeze(prediction_tensor).detach().cpu().numpy()
    ground_truth = torch.squeeze(ground_truth_tensor).detach().cpu().numpy()

    # pred_flat = prediction.flatten().reshape(-1,1)
    # gt_flat = ground_truth.flatten().reshape(-1,1)
    #
    # hd_distance = directed_hausdorff(pred_flat, gt_flat)[0]
    classes = np.unique(ground_truth)
    n = len(classes)
    hd95 = 0
    iou = 0
    dice = 0
    for i in classes:
        prediction_ = (prediction==i)
        ground_truth_ = (ground_truth==i)
        hd95 += binary.hd95(prediction_, ground_truth_, voxelspacing=None)

        intersection = np.logical_and(prediction_, ground_truth_)
        union = np.logical_or(prediction_, ground_truth_)
        iou += np.sum(intersection) / np.sum(union)
        dice += ((prediction_ * ground_truth_).sum() * 2) / (prediction_.sum() + ground_truth_.sum())
    return dice/n,hd95/n,iou/n
class Train_Test():
    def __init__(self, lr, epoch, save_path):
        setup_seed(66)
        # self.model = VNet().to(device)
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=1).to(device)
        # self.model.load_state_dict(torch.load('./PTH/seg+cls_1500.pth'))
        self.lr = lr
        self.epoch = epoch
        self.loss = Jonint_Loss().to(device)
        # self.t1loss = TI_Loss(dim=3, connectivity=26, inclusion=[], exclusion=[[1,2],[1,3],[1,4],[1,5],
        #                                                                        [2,3],[2,4],[2,5],
        #                                                                        [3,4],[3,5],[4,5]], min_thick=1)
        # 创建保存权重的路径
        self.model_path = os.path.join(save_path)
        # pth_path = "/media/dell/SATA1/ztfs-machine-learning-python/MC-Net-main/PTH1/seg+cls_500.pth"
        # checkpoint = torch.load(pth_path)
        # self.model.load_state_dict(checkpoint, strict=False)

    def Train(self, train_loader, val_loader):

        writer = SummaryWriter(self.model_path)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                     weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch)
        best_dice = 0
        best_epoch = 0
        iter_num = 0

        for epoch in range(0,self.epoch):
            print("=======Epoch:{}======Learning_rate:{}=========".format(epoch, optimizer.param_groups[0]['lr']))
            epoch_loss = 0
            epoch_dice = 0
            epoch_consist = 0
            # epoch_ti = 0
            self.model.train()
            tbar = tqdm.tqdm(train_loader)
            for batch_idx, sample in enumerate(tbar):  # 遍历的方式，提取了 image label;
                iter_num = iter_num + 1
                image = sample['image'].to(device)
                label = sample['label'].to(device)  # 输入
                assert label.sum()>500,'标签为空'
                outputs = self.model(image)

                num_outputs = len(outputs)

                y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
                y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape)

                loss_seg_dice = 0
                # ti_loss = 0
                for idx in range(num_outputs):
                    y = outputs[idx][:label_num, ...]

                    label_batch = torch.squeeze(label)
                    loss_seg_dice += F.cross_entropy(y, label_batch[:label_num])
                    loss_seg_dice += self.loss(y, label_batch[:label_num])
                    # ti_loss += 1e-4 *self.t1loss(y, label_batch[:label_num].unsqueeze(dim=1))

                    y_all = outputs[idx]
                    y_prob_all = F.softmax(y_all, dim=1)
                    y_ori[idx] = y_prob_all
                    y_pseudo_label[idx] = sharpening(y_prob_all)

                loss_consist = 0
                for i in range(num_outputs):
                    for j in range(num_outputs):
                        if i != j:
                            loss_consist += torch.mean((y_ori[i][label_num:] -
                                                        y_pseudo_label[j][label_num:]) ** 2)

                # consistency_weight = get_current_consistency_weight(iter_num // (len(tbar)*max_epoch // 40))
                consistency_weight = 1

                loss = loss_seg_dice + consistency_weight * loss_consist
                epoch_loss += loss.item()
                epoch_dice += loss_seg_dice.item()
                epoch_consist += loss_consist.item()
                # epoch_ti+= ti_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar('loss/loss', epoch_loss / len(tbar), epoch)
            writer.add_scalar('loss/loss_seg_dice', epoch_dice / len(tbar), epoch)
            writer.add_scalar('loss/loss_consist', epoch_consist / len(tbar), epoch)

            lr_scheduler.step()
            torch.cuda.empty_cache()
            if (epoch +1) % 10 == 0:
                dice_mean, labelmap = self.Val(val_loader)
                writer.add_image('img', np.expand_dims(labelmap[labelmap.shape[0] // 2], 0), epoch)
                if dice_mean >= best_dice:
                    best_epoch = epoch
                    best_dice = dice_mean
                    torch.save(self.model.state_dict(), self.model_path + '/seg_best.pth')
                open(self.model_path + '/' +'val_dice.txt', 'a') \
                    .write("epoch {} dice {} best_epoch {}\n".format(epoch, dice_mean, best_epoch))

            print('Finish Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch, epoch_loss / len(tbar)),
                  'dice_loss ', epoch_dice / len(tbar),
                  'loss_consist ', epoch_consist / len(tbar),
                  # 'loss_ti ', epoch_ti / len(tbar),
                  'best_epoch ', best_epoch)

            open(self.model_path + '/' +'loss.txt', 'a') \
                .write(
                "Epoch {} Loss {}  diceloss {}  consistloss {}  best_epoch {}\n".format(
                    epoch,
                    epoch_loss / len(tbar),
                    epoch_dice / len(tbar),
                    epoch_consist / len(tbar),
                    best_epoch))

            torch.save(self.model.state_dict(), self.model_path + '/seg_last.pth')
        writer.close()

    def Val(self, val_loader):
        patchdims = patch_size
        strides = [patchdims[0] // 2, patchdims[1] // 2, patchdims[2] // 2]
        tbar = tqdm.tqdm(val_loader)
        dice_all,hd_distance_all,iou_all = [],[],[]
        # pth_path = self.model_path + '/seg_last.pth'
        pth_path = self.model_path + '/seg+cls_3000.pth'
        checkpoint = torch.load(pth_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        with torch.no_grad():
            for i, (fname, image, label_seg) in enumerate(val_loader):
                fname = fname[0]
                vol_ori = sitk.ReadImage(preimgpath + '/' + fname)
                vol_ori = sitk.Image(vol_ori)
                # vol_origin = sitk.GetArrayFromImage(vol_ori)
                # 完成网络的前向传播
                image = image.to(device)

                leng, col, row = image.shape[2], image.shape[3], image.shape[4]
                score_map = torch.zeros(size=(6, leng, col, row), dtype=torch.float32).to(device)
                cnt = torch.zeros(size=(leng, col, row), dtype=torch.float32).to(device)

                patch = 0
                ###使用了逐patch进行分割的方式；
                slice = [i for i in range(0, leng, strides[0])]
                slice[-1] = leng - patchdims[0]
                slice_ = [i for i in slice if i <= slice[-1]]
                if slice_[-1] == slice_[-2]:
                    del slice_[-1]
                height = [i for i in range(0, col, strides[1])]
                height[-1] = col - patchdims[1]
                height_ = [i for i in height if i <= height[-1]]
                if height_[-1] == height_[-2]:
                    del height_[-1]
                width = [i for i in range(0, row, strides[2])]
                width[-1] = row - patchdims[2]
                width_ = [i for i in width if i <= width[-1]]
                if width_[-1] == width_[-2]:
                    del width_[-1]

                for i in slice_:
                    for j in height_:
                        for k in width_:
                            patch += 1
                            curpatch = image[:, :, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]]

                            predict_seg = self.model(curpatch)
                            predict_seg = F.softmax(predict_seg[0], dim=1)

                            curpatchoutlabel = torch.squeeze(predict_seg)

                            score_map[:, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]] += curpatchoutlabel
                            cnt[i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]] += 1

                ####--------------------------------------------------------
                labelmap0 = torch.zeros(size=[leng, col, row], dtype=torch.float32).to(device)
                # print(torch.unique(seg_map))
                seg_map = score_map / torch.unsqueeze(cnt, dim=0)
                for idx in range(0, leng):
                    curslicelabel = torch.squeeze(seg_map[:, idx, ].argmax(axis=0))  ##一个最大投票原则；
                    labelmap0[idx,] = curslicelabel

                dice_, hd_distance, iou = cal_metric(labelmap0, label_seg)
                dice_all.append(dice_)
                hd_distance_all.append(hd_distance)
                iou_all.append(iou)
                labelmap0 = labelmap0.data.cpu().numpy()
                labelmap = labelmap0.astype(np.uint8)
                outlabelmapraw = sitk.GetImageFromArray(labelmap.astype(np.uint8))
                outlabelmapraw.SetDirection(vol_ori.GetDirection())
                outlabelmapraw.SetSpacing(vol_ori.GetSpacing())
                outlabelmapraw.SetOrigin(vol_ori.GetOrigin())
                sitk.WriteImage(outlabelmapraw, os.path.join('/home/konata/Dataset/TAO_CT/ztf_CT/resnet+', fname))
                print(fname,dice_, hd_distance, iou)

                torch.cuda.empty_cache()
        dice_all = np.array(dice_all)
        dice_mean = np.mean(dice_all).item()
        hd_distance_all = np.array(hd_distance_all)
        hd_distance_mean = np.mean(hd_distance_all).item()
        iou_all = np.array(iou_all)
        iou_mean = np.mean(iou_all).item()
        print('dice:', dice_mean)
        print('95HD:', hd_distance_mean)
        print('miou:', iou_mean)
        return dice_mean, labelmap0

    def Test(self, test_loader):
        model = self.model
        pth_path = self.model_path + '/seg_last.pth'
        checkpoint = torch.load(pth_path)
        model.load_state_dict(checkpoint)
        patchdims = patch_size
        strides = [patchdims[0] // 2, patchdims[1] // 2, patchdims[2] // 2]
        tbar = tqdm.tqdm(test_loader)
        model.eval()
        with torch.no_grad():
            for i, (fname, image) in enumerate(tbar):
                try:
                    fname = fname[0]
                    vol_ori = sitk.ReadImage(testimgpath + '/' + fname)
                    vol_ori = sitk.Image(vol_ori)
                    # vol_origin = sitk.GetArrayFromImage(vol_ori)
                    # 完成网络的前向传播
                    image = image.to(device)

                    leng, col, row = image.shape[2], image.shape[3], image.shape[4]
                    # countermap = np.zeros((leng, col, row), dtype=np.float32)
                    score_map = torch.zeros(size=(6, leng, col, row), dtype=torch.float32).to(device)
                    cnt = torch.zeros(size=(leng, col, row), dtype=torch.float32).to(device)

                    patch = 0
                    ###使用了逐patch进行分割的方式；
                    slice = [i for i in range(0, leng, strides[0])]
                    slice[-1] = leng - patchdims[0]
                    slice_ = [i for i in slice if i <= slice[-1]]
                    if slice_[-1] == slice_[-2]:
                        del slice_[-1]
                    height = [i for i in range(0, col, strides[1])]
                    height[-1] = col - patchdims[1]
                    height_ = [i for i in height if i <= height[-1]]
                    if height_[-1] == height_[-2]:
                        del height_[-1]
                    width = [i for i in range(0, row, strides[2])]
                    width[-1] = row - patchdims[2]
                    width_ = [i for i in width if i <= width[-1]]
                    if width_[-1] == width_[-2]:
                        del width_[-1]

                    for i in slice_:
                        for j in height_:
                            for k in width_:
                                patch += 1
                                curpatch = image[:, :, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]]

                                predict_seg = self.model(curpatch)
                                predict_seg = F.softmax(predict_seg[0], dim=1)

                                curpatchoutlabel = torch.squeeze(predict_seg)

                                score_map[:, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]] += curpatchoutlabel
                                cnt[i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]] += 1
                    ####--------------------------------------------------------
                    labelmap0 = torch.zeros(size=[leng, col, row], dtype=torch.float32).to(device)
                    seg_map = score_map / torch.unsqueeze(cnt, dim=0)
                    for idx in range(0, leng):
                        curslicelabel = torch.squeeze(seg_map[:, idx, ].argmax(axis=0))  ##一个最大投票原则；
                        labelmap0[idx,] = curslicelabel
                    if len(torch.unique(labelmap0)) != 6:
                        print('error !6', fname,print(torch.unique(labelmap0)))

                    labelmap0 = labelmap0.data.cpu().numpy()
                    labelmap = labelmap0.astype(np.uint8)
                    outlabelmapraw = sitk.GetImageFromArray(labelmap.astype(np.uint8))
                    outlabelmapraw.SetDirection(vol_ori.GetDirection())
                    outlabelmapraw.SetSpacing(vol_ori.GetSpacing())
                    outlabelmapraw.SetOrigin(vol_ori.GetOrigin())
                    sitk.WriteImage(outlabelmapraw, os.path.join('/home/konata/Dataset/TAO_CT/ztf_CT/resnet+', fname))
                    torch.cuda.empty_cache()
                except:
                    print(fname)


def main(max_epoch_=1000,batch=5):
    global device
    global label_num
    global patch_size
    global preimgpath
    global testimgpath
    global max_epoch
    preimgpath = '/home/konata/Dataset/TAO_CT/ztf_CT/93CT/80origin'
    labelpath = '/home/konata/Dataset/TAO_CT/ztf_CT/93CT/80seg_6'
    testimgpath = '/home/konata/Dataset/TAO_CT/ztf_CT/origin50_1'
    traintxtpath = './MC_Net_main/1075.xlsx'
    path = './MC_Net_main/PTH6'
    # path = './PTH6'
    if not os.path.exists(path):
        os.mkdir(path)
    patch_size = (48, 128, 224)
    max_epoch = max_epoch_
    label_num = batch

    data_list = pd.read_excel(traintxtpath)
    val_list = data_list[80:93]
    train_list = data_list.drop([i for i in range(80, 93)])
    train_list = train_list.reset_index(drop=True)
    val_list = val_list.reset_index(drop=True)
    test_list = train_list[80:].reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    db_train = MySet(train_list,
                     preimgpath,
                     labelpath,
                     transform=transforms.Compose([
                         RandomCrop(patch_size),
                         ToTensor(),
                     ]))
    labelnum = 80
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(80, 1062))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, label_num, label_num)

    def worker_init_fn(worker_id):
        random.seed(666 + worker_id)

    train_loader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=6,
        pin_memory=True,
        worker_init_fn=worker_init_fn)
    # 根据batch size 提取了相应的数据。
    val_set = ValSet(val_list, preimgpath, labelpath)  ##能否在 这边定义新的构造函数。
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        num_workers=6,
        shuffle=False)  # 根据batch size 提取了相应的数据。

    test_set = TestSet(test_list, preimgpath=testimgpath)  ##能否在 这边定义新的构造函数。
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=0,
        shuffle=True)  # 根据batch size 提取了相应的数据。

    train_val = Train_Test(lr=1e-3, epoch=max_epoch, save_path=path)
    train_val.Train(train_loader, val_loader)
    dice_mean, labelmap0 = train_val.Val(val_loader)
    print(dice_mean)
    train_val.Test(test_loader)


if __name__ == '__main__':
    main()