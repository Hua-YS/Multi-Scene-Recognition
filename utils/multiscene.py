import csv
import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

ROOT = './data/'
SCENE_CATEGORY = ['Apron', 'BaseballField', 'BasketballField', 'Beach', 'Bridge', 'Cemetery', 'Commercial', 'Farmland', 'Woodland', 'GolfCourse', 'Greenhouse', 'Helipad', 'LakePond', 'OilFiled', 'Orchard', 'ParkingLot', 'Park', 'Pier', 'Port', 'Quarry', 'Railway', 'Residential', 'River', 'Roundabout', 'Runway', 'Soccer', 'SolarPannel', 'SparseShrub', 'Stadium', 'StorageTank', 'TennisCourt', 'TrainStation', 'WastewaterPlant', 'WindTrubine', 'Works', 'Sea']

def read_object_labels_csv(filename, only_gt=False, header=True):
    labels = []
    num_categories = len(SCENE_CATEGORY)
    print('[dataset] read', filename)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                name = row[0]
                gt = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                gt = torch.from_numpy(gt)
                item = (name, gt) if not only_gt else gt
                labels.append(item)
            rownum += 1
    return labels

class MultiScene(data.Dataset):
    def __init__(self, set, transform=None, target_transform=None):
        self.path_dataset = os.path.join(ROOT, 'MultiScene')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # define filename of csv file
        file_csv = os.path.join(self.path_dataset, set+'.csv')
        self.labels = read_object_labels_csv(file_csv)
    
        print('[dataset] MultiScene classification set=%s number of classes=%d  number of images=%d' % (
            set, len(SCENE_CATEGORY), len(self.labels)))

    def __getitem__(self, index):
        path, target = self.labels[index]
        img = Image.open(os.path.join(self.path_dataset, 'images', path+'.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path), target

    def __len__(self):
        return len(self.labels)

    def get_number_classes(self):
        return len(SCENE_CATEGORY)


#########################################################
# if taking scikit as the base
from skimage.feature import hog, local_binary_pattern

def MultiScene_sklearn(set):
    path_dataset = os.path.join(ROOT, 'MultiScene')
    path_feat = os.path.join(ROOT, 'MultiScene', 'feat'+set+'.npy')
    file_csv = os.path.join(path_dataset, set+'.csv')
    y = torch.stack(read_object_labels_csv(file_csv, True)).numpy()

    if os.path.isfile(path_feat):
        print('.npy files already exist.')
        x = np.load(path_feat)
        print('[dataset] MultiScene classification set=%s number of classes=%d  number of images=%d' % (set, len(SCENE_CATEGORY), len(y)))
        return x, y

    labels = read_object_labels_csv(file_csv)
    feat = np.zeros((len(labels), 2048+128))
    for index in range(len(labels)):
        print(index)
        path, target = labels[index]
        img = np.uint8(Image.open(os.path.join(path_dataset, 'images', path+'.jpg')).convert('RGB'))
        feat_hog32 = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), block_norm='L2')
        img_gray = np.uint8(Image.open(os.path.join(path_dataset, 'images', path+'.jpg')).convert('L'))
        feat_lbp16, _ = np.histogram(local_binary_pattern(img_gray, 16*8, 16, 'uniform'), density=True, bins=128, range=(0, 128))
        feat[index, :] = np.concatenate([feat_hog32, feat_lbp16])

    feat = np.array(feat)
    np.save(path_feat, feat)
    print('[dataset] MultiScene classification set=%s number of classes=%d  number of images=%d' % (set, len(SCENE_CATEGORY), len(y)))
    return feat, y





