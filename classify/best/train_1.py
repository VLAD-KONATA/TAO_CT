import os
import time

import SimpleITK as sitk
import pandas as pd
import torch.optim as optim
import tqdm
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from xgboost import XGBClassifier

from dataloader import SegSet
from dataset_4 import *

from resnet4 import *
from monai.visualize import GradCAM

writer = SummaryWriter('log')

def gradcam_show(data, heatmap,save_pth,title):
    data = data.cpu().detach().numpy()
    heatmap = heatmap.cpu().detach().numpy()
    plt.figure()
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(data, cmap='bone')

    plt.subplot(122)
    plt.axis('off')
    plt.imshow(data, cmap='bone')
    plt.imshow(heatmap, cmap='rainbow', alpha=0.3)
    plt.title(title)
    plt.savefig(save_pth)
    plt.close()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def suanscore(y_true, y_pred):
    # 自定义的f1 socre
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, recall, pre, f1


def Dice(seg_pre, seg_y):
    dice = ((seg_pre * seg_y).sum().item() * 2 + 1e-5) / (seg_pre.sum().item() + seg_y.sum().item() + 1e-5)
    return dice


def soft_dice(seg_pre, seg_y):
    dice = 0
    seg_y = torch.squeeze(seg_y).to(device)
    seg_pre = torch.squeeze(seg_pre).to(device)
    assert seg_pre.shape == seg_y.shape
    for i in range(1,6):
        seg_pre_ = (seg_pre == i).int()
        seg_y_ = (seg_y == i).int()
        dice += Dice(seg_pre_, seg_y_)
    return dice / 5

def data_augment(input):
    transform_compose = transforms.Compose([
            transforms.RandomRotation(degrees=45)
        ])

    image = input['image']
    label = input['label']
    image = torch.from_numpy(image).float()
    label = torch.from_numpy(label).long()
    image0 = input.reshape(image.shape[0] * image.shape[1], image.shape[2], image.shape[3])
    image0 = torch.unsqueeze(image0, dim=1)
    image0 = transform_compose(image0)
    image0 = image0.reshape(input.shape[0], input.shape[1], 128, 224)
    label0 = input.reshape(label.shape[0] * label.shape[1], label.shape[2], label.shape[3])
    label0 = torch.unsqueeze(label0, dim=1)
    label0 = transform_compose(label0)
    label0 = label0.reshape(input.shape[0], input.shape[1], 128, 224)

    input['image'] = image0
    input['label'] = label0
    return input

class Train_Test():
    def __init__(self, lr, epoch, save_path,fold=0,w=None,scale=None):
        setup_seed(66)
        # self.model = VNet().to(device)
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(),
                            self_adapt_w=False, default_w=w, n_input_channels=1, n_classes=2,scale=scale).to(device)
        self.model_name = 'seg+cls-seg50-{}-{}-'.format(w,scale)  #982seg
        # self.model_name = 'noqianyi-{}-{}-'.format(w,scale)  #"/media/dell/SATA1/ztfs-machine-learning-python/CT_SEG/PTH/seg+cls_3000.pth" resnet4.1
        self.fold = fold
        self.lr = lr
        self.epoch = epoch
        # 创建保存权重的路径
        self.model_path = os.path.join(save_path)

        pth_path = "/home/konata/Git/TAO_CT/MC_Net_main/PTH6/seg+cls_3000.pth" #5 6分割，使用resnet4
        checkpoint = torch.load(pth_path)
        self.model.load_state_dict(checkpoint, strict=False)


    def plot_figure(self, train_loss, train_acc, val_acc):
        index = self.fold
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(train_loss)), train_loss)
        plt.title('train_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.subplot(122)
        plt.plot(np.arange(len(train_acc)), train_acc, label='train_accuracy')
        plt.plot(np.arange(len(val_acc)), val_acc, label='val_accuracy')
        plt.legend()
        plt.title('train+val accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig(str(index) + '.png')
        # plt.show()

    def Train(self, train_loader, val_loader, seg_loader):

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                     weight_decay=0.005)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch)

        best_acc = 0
        best_epoch = 0
        trainacc=[]
        trainloss=[]
        valacc=[]
        scaler = GradScaler()
        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_dice = 0
            epoch_cls = 0
            correct_num = 0
            sample_num = []
            features = []
            names = []
            labels = []
            self.model.train(True)
            tbar = tqdm.tqdm(train_loader)
            for batch_idx, sample in enumerate(tbar):
                with autocast():
                    name = sample['name']
                    image = sample['image'].to(device)
                    seg_label = sample['label'].to(device)
                    cls_label = sample['cls_label'].to(device)
                    sample_num.extend(list(cls_label.cpu().detach().numpy()))
                    assert seg_label.sum() > 2000, '未定位'
                    for iter_ in range(1):
                        if iter_==0:
                            x_cls, x_seg, loss, every_loss, feature = self.model(image, seg_label, cls_label)
                        else:
                            x_cls, x_seg, loss, every_loss, feature = self.model(image1, seg_label, cls_label)
                        seg_1 = F.softmax(x_seg, dim=1)
                        seg_1 = seg_1.argmax(1).float()
                        seg_1 = torch.unsqueeze(seg_1, dim=1)
                        image1 = image+seg_1
                        image1 = (image1 - torch.mean(image1)) / torch.std(image1)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        # optimizer.zero_grad()
                        # loss.backward()
                        # optimizer.step()
                    features.append(feature)
                    names.extend(name)
                    labels.append(cls_label)

                labels_predict = x_cls.argmax(1)
                correct_num += len(torch.where(cls_label == labels_predict)[0])

                epoch_loss += loss.item()
                epoch_dice +=every_loss[0].item()
                epoch_cls += every_loss[1].item()

            train_acc = correct_num/len(sample_num)
            train_loss = epoch_loss / len(tbar)
            trainacc.append(train_acc)
            trainloss.append(train_loss)
            writer.add_scalar('acc/train', train_acc, epoch)
            writer.add_scalar('loss/Loss', train_loss, epoch)
            writer.add_scalar('loss/DiceLoss', epoch_dice / len(tbar), epoch)
            writer.add_scalar('loss/ClsLoss', epoch_cls / len(tbar), epoch)

            lr_scheduler.step()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            features = torch.cat(features, dim=0).detach().cpu().numpy()
            labels = torch.cat(labels, dim=0).detach().cpu().numpy().reshape((len(names), 1))
            names = np.array(names).reshape((len(names), 1))
            features = np.concatenate((names, labels, features), axis=1)

            val_acc, valfeature= self.Val(val_loader,epoch)
            val_acc = round(val_acc,3)
            valacc.append(val_acc)
            writer.add_scalar('acc/val', val_acc, epoch)

            if val_acc >= best_acc and self.epoch>1:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.model_path + self.model_name+'best_fold%d.pth'%(self.fold))

            dice_mean, labelmap = self.Test_seg(seg_loader)
            writer.add_image('img/val', np.expand_dims(labelmap[labelmap.shape[0] // 2], 0), epoch)
            writer.add_image('img/train', seg_label[-1][:,25,:,:], epoch)

            open('val_dice_pth4.txt', 'a') \
                .write("epoch {} dice {} train_acc {} val_acc {} best_epoch {} best_acc {} \n".format(
                epoch, dice_mean, train_acc, val_acc, best_epoch, best_acc))
            print('dice: ', dice_mean, 'train_acc', train_acc, 'val_acc', val_acc)

            print('Finish Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch, epoch_loss / len(tbar)),
                  'best_epoch ', best_epoch,
                  'best_acc', best_acc
                  )

            open('LogFusionTraining_pth4.txt', 'a') \
                .write(
                "Epoch {} Loss {} DiceLoss {} ClsLoss {} best_epoch {}\n".format(
                    epoch,
                    epoch_loss / len(tbar),
                    epoch_dice / len(tbar),
                    epoch_cls / len(tbar),
                    best_epoch))
            if self.epoch>1:
                torch.save(self.model.state_dict(), self.model_path + self.model_name+'{}_last.pth'.format(self.fold))
        if self.epoch>1:
            self.plot_figure(trainloss,trainacc,valacc)
        return features

    def Val(self, valid_loader,epoch):
        # pth_path = self.model_path + 'seg+cls{}_{}.pth'.format(self.fold, epoch + 1)
        # checkpoint = torch.load(pth_path)
        # self.model.load_state_dict(checkpoint)
        patchdims = patch_size
        strides = [patchdims[0] // 2, patchdims[1] // 2, patchdims[2] // 2]
        self.model.eval()
        tbar = tqdm.tqdm(valid_loader)
        y_true = []
        y_pred = []
        features = []
        names = []
        labels = []

        with torch.no_grad():
            for i, sample in enumerate(tbar):
                # 完成网络的前向传播
                name = sample['name']
                names.append(name)
                image = sample['image'].to(device)
                seg_label = sample['label'].to(device)
                cls_label = sample['cls_label'].to(device)
                labels.append(cls_label)

                pre = 0
                feature = 0
                num = 0

                leng, col, row = image.shape[2], image.shape[3], image.shape[4]
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
                            imgpatch = image[:, :, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]]
                            segpatch = seg_label[:, :, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]]
                            x_cls, x_seg, loss, every_loss, feature_ = self.model(imgpatch, segpatch, cls_label,if_train=False)
                            pre += x_cls
                            feature += feature_
                            num += 1
                feature = feature / num
                features.append(feature)
                pre = pre.argmax(1)
                labels_predict = pre.item()

                y_pred.append(labels_predict)
                y_true.append(cls_label.item())
        features = torch.cat(features, dim=0).detach().cpu().numpy()
        labels = torch.cat(labels, dim=0).detach().cpu().numpy().reshape((len(names), 1))
        names = np.array(names).reshape((len(names), 1))
        features = np.concatenate((names, labels, features), axis=1)
        acc, recall, pre, f1 = suanscore(y_true, y_pred)
        return acc, features

    def Test(self, test_loader,seg_loader):
        patchdims = patch_size
        strides = [patchdims[0] // 2, patchdims[1] // 2, patchdims[2] // 2]
        pth_path = self.model_path + self.model_name+'best_fold%d.pth'%(self.fold)
        checkpoint = torch.load(pth_path)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        tbar = tqdm.tqdm(test_loader)
        y_true = []
        y_pred = []
        error_name = []
        features = []
        names = []
        labels = []
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                # 完成网络的前向传播
                image = sample['image'].to(device)
                seg_label = sample['label'].to(device)
                cls_label = sample['cls_label'].to(device)
                labels.append(cls_label)
                fnames = sample['name']
                names.append(fnames)

                pre = 0
                num = 0
                feature = 0

                leng, col, row = image.shape[2], image.shape[3], image.shape[4]
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
                            imgpatch = image[:, :, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]]
                            segpatch = seg_label[:, :, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]]
                            x_cls, x_seg, loss, every_loss, feature_ = self.model(imgpatch, segpatch, cls_label,if_train=False)

                            pre += x_cls
                            feature += feature_
                            num += 1

                feature = feature / num
                features.append(feature)
                pre = pre.argmax(1)
                labels_predict = pre.item()

                if labels_predict != cls_label:
                    error_name.append(fnames)
                y_pred.append(labels_predict)
                y_true.append(cls_label.item())

        acc, recall, pre, f1 = suanscore(y_true, y_pred)

        dice_mean, labelmap0 = self.Test_seg(seg_loader)
        assess_data = [dice_mean, acc, recall, pre, f1]
        print('dice_mean: ',dice_mean)
        features = torch.cat(features, dim=0).detach().cpu().numpy()
        labels = torch.cat(labels, dim=0).detach().cpu().numpy().reshape((len(names), 1))
        names = np.array(names).reshape((len(names), 1))
        features = np.concatenate((names, labels, features), axis=1)
        return assess_data, error_name, features

    def Test_seg(self, seg_loader):
        patchdims = patch_size
        strides = [patchdims[0] // 2, patchdims[1] // 2, patchdims[2] // 2]
        tbar = tqdm.tqdm(seg_loader)
        dice_all = []
        self.model.eval()
        with torch.no_grad():
            for i, (fname, image, label_seg) in enumerate(tbar):
                fname = fname[0]
                vol_ori = sitk.ReadImage(preimgpath + '/' + fname)
                vol_ori = sitk.Image(vol_ori)
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

                            predict_seg = self.model(x=curpatch, if_train=False)
                            predict_seg = F.softmax(predict_seg[1], dim=1)

                            curpatchoutlabel = torch.squeeze(predict_seg)

                            score_map[:, i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]] += curpatchoutlabel
                            cnt[i:i + patchdims[0], j:j + patchdims[1], k:k + patchdims[2]] += 1

                seg_map = score_map / torch.unsqueeze(cnt, dim=0)
                labelmap0 = torch.zeros(size=[leng, col, row], dtype=torch.float32).to(device)
                try:
                    for idx in range(0, leng):
                        curslicelabel = torch.squeeze(seg_map[:, idx, ].argmax(axis=0))  ##一个最大投票原则；
                        labelmap0[idx,] = curslicelabel
                except:
                    print('error', fname)

                dice_ = soft_dice(labelmap0, label_seg)
                dice_all.append(dice_)
                labelmap0 = labelmap0.data.cpu().numpy()
                labelmap = labelmap0.astype(np.uint8)
                outlabelmapraw = sitk.GetImageFromArray(labelmap.astype(np.uint8))
                outlabelmapraw.SetDirection(vol_ori.GetDirection())
                outlabelmapraw.SetSpacing(vol_ori.GetSpacing())
                outlabelmapraw.SetOrigin(vol_ori.GetOrigin())
                sitk.WriteImage(outlabelmapraw, os.path.join('/home/konata/Git/TAO_CT/thesis/best/seg/', fname))
                torch.cuda.empty_cache()
        dice_all = np.array(dice_all)
        dice_mean = np.mean(dice_all).item()
        return dice_mean, labelmap0


def main(w=0.2,scale=3,epoch=150):
    print('*' * 50 + 'w=' + str(w) + '-' + 'scale=' + str(scale) + '*' * 50 + '\n', )
    global path
    global max_epoch
    global patch_size
    global preimgpath
    global device
    preimgpath = '/home/konata/Dataset/TAO_CT/ztf_CT/93CT/80origin'
    labelpath = '/home/konata/Dataset/TAO_CT/ztf_CT/93CT/80seg_6'
    path = '/home/konata/Git/TAO_CT/thesis/best/7pth/'
    patch_size = (48, 128, 224)
    max_epoch = epoch

    data_list = pd.read_excel('/home/konata/Dataset/TAO_CT/ztf_CT/93.xlsx')[0:93]
    feature_zuxue = pd.read_excel('/home/konata/Dataset/TAO_CT/ztf_CT/982.xlsx')
    data = feature_zuxue.iloc[:, 0:2]
    seg_list = data_list[80:93]
    seg_list = seg_list.reset_index(drop=True)
    error_ = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_assess = []  # 存放CNN分类结果

    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    for index, (tra_index, test_index) in enumerate(skf.split(data['id'].values, data['label'].values)):
        print('========== {} =========='.format(index + 1))
        train_ = data.iloc[tra_index]
        test = data.iloc[test_index]

        train, val = train_test_split(train_, test_size=0.1, random_state=666, stratify=train_['label'])

        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)

        train['label'] = train['label'].astype(int)
        val['label'] = val['label'].astype(int)
        test['label'] = test['label'].astype(int)
        print(train.shape)
        db_train = MySet(train,
                         preimgpath,
                         labelpath,
                         transform=transforms.Compose([
                             RandomCrop(patch_size),
                             ToTensor(),
                             RandomRot(),
                         ]))

        def worker_init_fn(worker_id):
            random.seed(666 + worker_id)


        train_loader = DataLoader(
            db_train,
            batch_size=12,
            num_workers=6,
            pin_memory=True,
            worker_init_fn=worker_init_fn)

        seg_set = SegSet(seg_list, preimgpath, labelpath)
        seg_loader = DataLoader(
            seg_set,
            batch_size=1,
            num_workers=6,
            shuffle=True)

        db_val = MySet(val,
                         preimgpath,
                         labelpath,
                         transform=transforms.Compose([
                             # RandomCrop((48,144,240)),
                             ToTensor(),
                             # RandomRot(),
                         ]))
        val_loader = DataLoader(
            db_val,
            batch_size=1,
            num_workers=6,
            shuffle=True)
        db_test = MySet(test,
                         preimgpath,
                         labelpath,
                         transform=transforms.Compose([
                             # RandomCrop((48,144,240)),
                             ToTensor(),
                             # RandomRot(),
                         ]))
        test_loader = DataLoader(
            db_test,
            batch_size=1,
            num_workers=6,
            shuffle=True)

        time0 = time.time()
        train_val = Train_Test(lr=1e-4, epoch=max_epoch, save_path=path, fold=index,w=w,scale=scale)
        feature_train = train_val.Train(train_loader, val_loader, seg_loader)
        feature_train = pd.DataFrame(feature_train)

        time1 = time.time()
        print('训练时间：{:.2f}分钟'.format((time1 - time0) / 60))
        assess_data, error_name, feature_test = train_val.Test(test_loader,seg_loader)
        feature_test = pd.DataFrame(feature_test)
        for name in error_name:
            name = name[0]
            error_[name] = error_.get(name, 0) + 1

        test_assess.append(assess_data)
        print("测试集准确率为：", assess_data[1])
        print("测试集召回率为：", assess_data[2])
        print("测试集精确度为：", assess_data[3])
        print("测试集f1分数为：", assess_data[4])
        time2 = time.time()
        print('测试时间：{:.2f}秒'.format((time2 - time1)))
        writer.close()
       
    test_assess = np.array(test_assess).reshape((-1, 5))
    mean_assess = np.mean(test_assess, axis=0)
    mean_dice, mean_acc, mean_recall, mean_pre, mean_f1 = mean_assess[0], mean_assess[1], mean_assess[2], mean_assess[3], mean_assess[4]
    print('cnn五折交叉：')
    print('dice', test_assess[:, 0])
    print('acc', test_assess[:, 1])
    print('dice: ',mean_dice,'acc：', mean_acc, 'recall:', mean_recall, 'pre:', mean_pre, 'f1:', mean_f1)
   

if __name__ == '__main__':
    acc = main()