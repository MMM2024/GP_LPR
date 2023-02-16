from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import lmdb
import os
import torch
from skimage.transform import resize
import utils
import params_LPR as params
from PIL import Image

import random

from straug.warp import *
from straug.geometry import *
from straug.pattern import *
from straug.noise import *
# from straug.weather import *
from straug.camera import *
from straug.process import *



class LPR_LMDB_Dataset(Dataset):

    def __init__(self, root, alphabets=params.alphabet, isCCPD=False, K=8, use_ratio=1.0, isdegrade=False, isAug=False):

        self.root = root
        # self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)


        self.isCCPD = isCCPD

        self.isdegrade = isdegrade
        self.K = K
        self.alphabets = alphabets
        self.isAug = isAug
        
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = int(nSamples * use_ratio)
            self.filtered_index_list = [index + 1 for index in range(self.nSamples)]


    def __len__(self):
        return self.nSamples

    
    def Image2Numpy(self, pil_image):
        np_image = np.array(pil_image).astype("float32")
        np_image = np_image/255
        np_image = np_image.reshape(1, np_image.shape[0], np_image.shape[1])

        return np_image


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            if self.isCCPD == True:
                label = list(map(int, label.split('_'))) #convert str list to int list
                for i in range(params.K-len(label)):
                    label.append(-1)
                label = torch.LongTensor(label) + 1

            img_key = 'image-%09d'.encode() % index
            image = txn.get(img_key)
            image_array = np.frombuffer(image, dtype=np.float32)

            image_array = image_array.reshape(1, 32, 96)
            image_array = np.around(image_array, decimals=4)
            if self.isAug:
                image_array = image_array.reshape(32, 96)*255
                pil_img = Image.fromarray(image_array.astype("uint8"), mode="L")

                aug_img = augment(pil_img)
                np_aug_img = self.Image2Numpy(aug_img)

                return (np_aug_img, label)


            if self.isdegrade:
                if random.random() < 0.05:
                    image_array = image_array.reshape(32, 96)
                    image_array = cv2.resize(image_array,(48,16))
                    # print('image_array',image_array.shape)            

                    image_array = cv2.resize(image_array,(96,32))   
                    image_array = (np.reshape(image_array, (32,96, 1))).transpose(2, 0, 1)


            return (image_array, label)



def Perspective(img, mag=-1, prob=1.):
        rng = np.random.default_rng()

        if rng.uniform(0, 1) > prob:
            return img

        w, h = img.size

        # upper-left, upper-right, lower-left, lower-right
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        # low = 0.3

        # b = [.05, .1, .15]
        b = [.1, .15, .2]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        low = b[index]

        high = 1 - low
        
        if rng.uniform(0, 1) > 0.5:
            topright_y = rng.uniform(low, low + .1) * h
            topright_x = rng.uniform(high+low/2, 1) * w
            bottomleft_x = rng.uniform(0, low/2) * w
            bottomleft_y = rng.uniform(high - .1, high) * h
            dest = np.float32([[0, 0], [topright_x, topright_y], [bottomleft_x, bottomleft_y], [w, h]])
        else:
            topleft_y = rng.uniform(low, low + .1) * h
            topleft_x = rng.uniform(0, low/2) * w
            bottomright_y = rng.uniform(high - .1, high) * h
            bottomright_x = rng.uniform(high+low/2, 1) * w
            dest = np.float32([[topleft_x, topleft_y], [w, 0], [0, h], [bottomright_x, bottomright_y]])

        M = cv2.getPerspectiveTransform(src, dest)
        img = np.asarray(img)
        img = cv2.warpPerspective(img, M, (w, h))
        img = Image.fromarray(img)

        return img


def augment(PIL_img, isPIL=True, imgW=96, imgH=32):
    noise_switch = {
                "0": GaussianNoise(),
                "1": SpeckleNoise()}

    camera_switch = {
                "0": Contrast(),
                "1": Brightness(),
                "2": JpegCompression(),
                "3": Pixelate()}
    process_switch = {
                "0": Posterize(),
                "1": Equalize(),
                "2": AutoContrast(),
                "3": Sharpness(),
                "4": Color(),}  


    if isPIL:
        img = PIL_img
    else:
        img = Image.fromarray(PIL_img) #转为PIL Image格式

    switch = random.randint(0, 2)
    # switch = 5

    if switch == 0:
        img = noise_switch[str(random.randint(0,1))](img, mag=0)
    elif switch == 1:
        img = camera_switch[str(random.randint(0,3))](img, mag=0)
    elif switch == 2:

        img = process_switch[str(random.randint(0,4))](img, mag=0)

    # img = Perspective(img, mag=0,  prob=0.5)
    img = Perspective(img, mag=2,  prob=0.5)
    # img = Rotate()(img, mag=0,  prob=0.4)

    if not isPIL:
        img = np.asarray(img)

    return img

if __name__ == '__main__':
    dataset = LPR_LMDB_Dataset("/path/to/lmdb_dataset", isAug=True)
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    for i_batch, (image, label) in enumerate(dataloader):
        aug_img = torch.reshape(image[0], (32, 96))  
        aug_img = aug_img.numpy()*255
        cv2.imwrite('rb3_aug3.png', aug_img)