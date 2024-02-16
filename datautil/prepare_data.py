import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from datautil.datasplit import getdataloader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
import pickle
from torch.utils.data import Dataset

def get_data(data_name):
    """Return the algorithm class with the given name."""
    datalist = {'officehome': 'img_union', 'pacs': 'img_union', 'vlcs': 'img_union', 'medmnist': 'medmnist',
                'medmnistA': 'medmnist', 'medmnistC': 'medmnist', 'pamap': 'pamap', 'ShapeNetPart': 'ShapeNetPart'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]

def gettransforms():
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test

def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))

def load_data_partseg(partition):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    # if partition == 'trainval':
    #     file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
    #            + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    # else:
    #     file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))

    if partition == 'train01':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train01.h5'))
    elif partition == 'train02':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train02.h5'))
    elif partition == 'train03':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train03.h5'))
    elif partition == 'train04':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train04.h5'))
    # elif partition == 'train05':
    #     file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train05.h5'))
    # elif partition == 'train06':
    #     file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train06.h5'))
    elif partition == 'test01':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*test01.h5'))
    elif partition == 'test02':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*test02.h5'))
    elif partition == 'test03':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*test03.h5'))
    elif partition == 'test04':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*test04.h5'))
    # elif partition == 'test05':
    #     file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*test05.h5'))
    # elif partition == 'test06':
    #     file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*test06.h5'))
    elif partition == 'val01':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val01.h5'))
    elif partition == 'val02':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val02.h5'))
    elif partition == 'val03':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val03.h5'))
    elif partition == 'val04':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val04.h5'))
    # elif partition == 'val05':
    #     file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val05.h5'))
    # elif partition == 'val06':
    #     file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val06.h5'))

    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg

def load_color_partseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/partseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    partseg_colors = np.array(colors)
    partseg_colors = partseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1350
    img = np.zeros((1350, 1890, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
    column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    column_gaps = [320, 320, 300, 300, 285, 285]
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for row in range(0, img_size):
        column_index = 32
        for column in range(0, img_size):
            color = partseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.76, (0, 0, 0), 2)
            column_index = column_index + column_gaps[column]
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 50:
                cv2.imwrite("prepare_data/meta/partseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column + 1 >= column_numbers[row]):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


class ShapeNetPartDataset(Dataset):
    def __init__(self, num_points, partition, class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'human': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [10, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 10, 12, 14, 18, 22, 25, 28, 30, 34, 36, 42, 44, 47, 50, 53]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice
        self.partseg_colors = load_color_partseg()

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 56
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train*':        #注意这里要改
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class mydataset(object):
    def __init__(self, args):
        self.x = None
        self.targets = None
        self.dataset = None
        self.transform = None
        self.target_transform = None
        self.loader = None
        self.args = args

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        x = self.input_trans(self.loader(self.x[index]))
        ctarget = self.target_trans(self.targets[index])
        return x, ctarget

    def __len__(self):
        return len(self.targets)


class ImageDataset(mydataset):
    def __init__(self, args, dataset, root_dir, domain_name):
        super(ImageDataset, self).__init__(args)
        self.imgs = ImageFolder(root_dir+domain_name).imgs
        self.domain_num = 0
        self.dataset = dataset
        imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.targets = np.array(labels)
        transform, _ = gettransforms()
        target_transform = None
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.pathx = imgs
        self.x = self.pathx


class MedMnistDataset(Dataset):
    def __init__(self, filename='', transform=None):
        self.data = np.load(filename+'xdata.npy')
        self.targets = np.load(filename+'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = transform

        self.data = torch.Tensor(self.data)
        self.data = torch.unsqueeze(self.data, dim=1)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class PamapDataset(Dataset):
    def __init__(self, filename='../data/pamap/', transform=None):
        self.data = np.load(filename+'x.npy')
        self.targets = np.load(filename+'y.npy')
        self.select_class()
        self.transform = transform
        self.data = torch.unsqueeze(torch.Tensor(self.data), dim=1)
        self.data = torch.einsum('bxyz->bzxy', self.data)

    def select_class(self):
        xiaochuclass = [0, 5, 12]
        index = []
        for ic in xiaochuclass:
            index.append(np.where(self.targets == ic)[0])
        index = np.hstack(index)
        allindex = np.arange(len(self.targets))
        allindex = np.delete(allindex, index)
        self.targets = self.targets[allindex]
        self.data = self.data[allindex]
        ry = np.unique(self.targets)
        ry2 = {}
        for i in range(len(ry)):
            ry2[ry[i]] = i
        for i in range(len(self.targets)):
            self.targets[i] = ry2[self.targets[i]]

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CovidDataset(Dataset):
    def __init__(self, filename='../data/covid19/', transform=None):
        self.data = np.load(filename+'xdata.npy')
        self.targets = np.load(filename+'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = transform
        self.data = torch.Tensor(self.data)
        self.data = torch.einsum('bxyz->bzxy', self.data)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def getfeadataloader(args):
    trl, val, tel = [], [], []
    trd, vad, ted = [], [], []
    for item in args.domains:
        data = ImageDataset(args, args.dataset,
                            args.root_dir+args.dataset+'/', item)
        l = len(data)
        index = np.arange(l)
        np.random.seed(args.seed)
        np.random.shuffle(index)
        l1, l2, l3 = int(l*args.datapercent), int(l *
                                                  args.datapercent), int(l*0.2)
        trl.append(torch.utils.data.Subset(data, index[:l1]))
        val.append(torch.utils.data.Subset(data, index[l1:l1+l2]))
        tel.append(torch.utils.data.Subset(data, index[l1+l2:l1+l2+l3]))
        _, target_transform = gettransforms()
        val[-1].transform = target_transform
        tel[-1].transform = target_transform
        trd.append(torch.utils.data.DataLoader(
            trl[-1], batch_size=args.batch, shuffle=True))
        vad.append(torch.utils.data.DataLoader(
            val[-1], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[-1], batch_size=args.batch, shuffle=False))
    return trd, vad, ted


def img_union(args):
    return getfeadataloader(args)


def getlabeldataloader(args, data):
    trl, val, tel = [], [], []
    trd, vad, ted = [], [], []
    trl.append(ShapeNetPartDataset(partition='train01', num_points=args.num_points, class_choice=args.class_choice))
    trl.append(ShapeNetPartDataset(partition='train02', num_points=args.num_points, class_choice=args.class_choice))
    trl.append(ShapeNetPartDataset(partition='train03', num_points=args.num_points, class_choice=args.class_choice))
    trl.append(ShapeNetPartDataset(partition='train04', num_points=args.num_points, class_choice=args.class_choice))
    # trl.append(ShapeNetPartDataset(partition='train05', num_points=args.num_points, class_choice=args.class_choice))
    # trl.append(ShapeNetPartDataset(partition='train06', num_points=args.num_points, class_choice=args.class_choice))
    tel.append(ShapeNetPartDataset(partition='test01', num_points=args.num_points, class_choice=args.class_choice))
    tel.append(ShapeNetPartDataset(partition='test02', num_points=args.num_points, class_choice=args.class_choice))
    tel.append(ShapeNetPartDataset(partition='test03', num_points=args.num_points, class_choice=args.class_choice))
    tel.append(ShapeNetPartDataset(partition='test04', num_points=args.num_points, class_choice=args.class_choice))
    # tel.append(ShapeNetPartDataset(partition='test05', num_points=args.num_points, class_choice=args.class_choice))
    # tel.append(ShapeNetPartDataset(partition='test06', num_points=args.num_points, class_choice=args.class_choice))
    val.append(ShapeNetPartDataset(partition='val01', num_points=args.num_points, class_choice=args.class_choice))
    val.append(ShapeNetPartDataset(partition='val02', num_points=args.num_points, class_choice=args.class_choice))
    val.append(ShapeNetPartDataset(partition='val03', num_points=args.num_points, class_choice=args.class_choice))
    val.append(ShapeNetPartDataset(partition='val04', num_points=args.num_points, class_choice=args.class_choice))
    # val.append(ShapeNetPartDataset(partition='val05', num_points=args.num_points, class_choice=args.class_choice))
    # val.append(ShapeNetPartDataset(partition='val06', num_points=args.num_points, class_choice=args.class_choice))

    for i in range(len(trl)):
        trd.append(torch.utils.data.DataLoader(
            trl[i], batch_size=args.batch, shuffle=True))
        vad.append(torch.utils.data.DataLoader(
            val[i], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[i], batch_size=args.batch, shuffle=False))

    return trd, vad, ted


def medmnist(args):
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 11
    return trd, vad, ted


def pamap(args):
    data = PamapDataset(args.root_dir+'pamap/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 10
    return trd, vad, ted


def covid(args):
    data = CovidDataset(args.root_dir+'covid19/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 4
    return trd, vad, ted


class combinedataset(mydataset):
    def __init__(self, datal, args):
        super(combinedataset, self).__init__(args)

        self.x = np.hstack([np.array(item.x) for item in datal])
        self.targets = np.hstack([item.targets for item in datal])
        s = ''
        for item in datal:
            s += item.dataset+'-'
        s = s[:-1]
        self.dataset = s
        self.transform = datal[0].transform
        self.target_transform = datal[0].target_transform
        self.loader = datal[0].loader


def getwholedataset(args):
    datal = []
    for item in args.domains:
        datal.append(ImageDataset(args, args.dataset,
                     args.root_dir+args.dataset+'/', item))
    # data=torch.utils.data.ConcatDataset(datal)
    data = combinedataset(datal, args)
    return data


def img_union_w(args):
    return getwholedataset(args)


def medmnist_w(args):
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    args.num_classes = 11
    return data


def pamap_w(args):
    data = PamapDataset(args.root_dir+'pamap/')
    args.num_classes = 10
    return data

"num_points 是arg传参值 默认2048"
def ShapeNetPart(args):
    trl, val, tel = [], [], []
    trd, vad, ted = [], [], []
    trl.append(ShapeNetPartDataset(partition='train01', num_points=args.num_points, class_choice=args.class_choice))
    trl.append(ShapeNetPartDataset(partition='train02', num_points=args.num_points, class_choice=args.class_choice))
    trl.append(ShapeNetPartDataset(partition='train03', num_points=args.num_points, class_choice=args.class_choice))
    trl.append(ShapeNetPartDataset(partition='train04', num_points=args.num_points, class_choice=args.class_choice))
    # trl.append(ShapeNetPartDataset(partition='train05', num_points=args.num_points, class_choice=args.class_choice))
    # trl.append(ShapeNetPartDataset(partition='train06', num_points=args.num_points, class_choice=args.class_choice))
    tel.append(ShapeNetPartDataset(partition='test01', num_points=args.num_points, class_choice=args.class_choice))
    tel.append(ShapeNetPartDataset(partition='test02', num_points=args.num_points, class_choice=args.class_choice))
    tel.append(ShapeNetPartDataset(partition='test03', num_points=args.num_points, class_choice=args.class_choice))
    tel.append(ShapeNetPartDataset(partition='test04', num_points=args.num_points, class_choice=args.class_choice))
    # tel.append(ShapeNetPartDataset(partition='test05', num_points=args.num_points, class_choice=args.class_choice))
    # tel.append(ShapeNetPartDataset(partition='test06', num_points=args.num_points, class_choice=args.class_choice))
    val.append(ShapeNetPartDataset(partition='val01', num_points=args.num_points, class_choice=args.class_choice))
    val.append(ShapeNetPartDataset(partition='val02', num_points=args.num_points, class_choice=args.class_choice))
    val.append(ShapeNetPartDataset(partition='val03', num_points=args.num_points, class_choice=args.class_choice))
    val.append(ShapeNetPartDataset(partition='val04', num_points=args.num_points, class_choice=args.class_choice))
    # val.append(ShapeNetPartDataset(partition='val05', num_points=args.num_points, class_choice=args.class_choice))
    # val.append(ShapeNetPartDataset(partition='val06', num_points=args.num_points, class_choice=args.class_choice))

    for i in range(len(trl)):
        trd.append(torch.utils.data.DataLoader(
            trl[i], batch_size=args.batch, shuffle=True))
        vad.append(torch.utils.data.DataLoader(
            val[i], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[i], batch_size=args.batch, shuffle=False))

    args.num_classes = 10

    return trd, vad, ted


def get_whole_dataset(data_name):
    datalist = {'officehome': 'img_union_w', 'pacs': 'img_union_w', 'vlcs': 'img_union_w', 'medmnist': 'medmnist_w',
                'medmnistA': 'medmnist_w', 'medmnistC': 'medmnist_w', 'pamap': 'pamap_w', 'ShapeNetPart': 'ShapeNetPart_w'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]
if __name__ == '__main__':
    trainval = ShapeNetPart(2048, 'trainval')
    test = ShapeNetPart(2048, 'test')
    data, label, seg = trainval[0]
    print(data.shape)
    print(label.shape)
    print(seg.shape)