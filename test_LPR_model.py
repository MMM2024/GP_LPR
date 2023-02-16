import numpy as np
import sys, os
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import params_test as params

from dataset_own_lmdb import LPR_LMDB_Dataset
import models.LPR_model as model



import Levenshtein
from matplotlib import pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
import seaborn as sns

str1 = params.alphabet
alphabet = params.alphabet

alphabets_dict = []
alphabets_dict.append('-')
for i in range(len(params.alphabet)):
    alphabets_dict.append(params.alphabet[i])

nclass = len(alphabet)+1
torch.cuda.set_device(params.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize_feature(temp):
    features = temp.detach().cpu().numpy()
    feature = features[0].reshape(-1, features.shape[-1])
    feature_norm = (feature-np.min(feature))/(np.max(feature)-np.min(feature))*255
    cv2.imwrite('feature_images.jpg',feature_norm)


def val(lpr_model, val_loader):

    print('Start val')
    lpr_model.eval()

    n_correct = 0
    n_woC_correct, n_C_correct = 0,0
    all_predict = []
    infer_time = 0
    confuse_matrix = np.zeros((nclass,nclass))
    char_acc = np.zeros((params.K))
    char_acc_num = 0
    ed1 = 0
    wrong_pred = []
    start_time_1 = time.time()
    forward_time_1 = time.time() - start_time_1

    for i_batch, (image, label) in enumerate(val_loader):
        
        image = image.cuda()
        start_time = time.time()
        atten_list, preds, enc_slf_attn = lpr_model(image)

        forward_time = time.time() - start_time
        v_fc_weights = lpr_model.state_dict()['attention.fc_v.weight']
        
        batch_size = image.size(0)
        cost = 0
        _, preds_index = preds.max(2)
        sim_preds = converter.decode_sa(preds_index)

        infer_time += forward_time


        all_predict += sim_preds

        for  pred, target in zip(sim_preds, label):
            all_predict.append(pred)
            if pred == target:
                n_correct += 1
            else:
                wrong_pred.append((i_batch, target, pred))
                print('i_batch:', i_batch, '  target:', target, '  pred:', pred)


            lev_distance = Levenshtein.distance(pred,target)

            if lev_distance <=1:
                ed1 += 1

            if pred[0] == target[0]:
                n_C_correct +=1
            
            if pred[1:] == target[1:]:
                n_woC_correct +=1
            if len(pred) == len(target):
                char_acc_num += 1

                for i in range(len(target)):
                    confuse_matrix[converter.encode_char(pred[i]),converter.encode_char(target[i])] += 1
                    if pred[i] == target[i]:
                        char_acc[i] +=1
        

    acc = float(n_correct)/len(val_dataset)
    print('time: ', float(infer_time)/len(val_dataset)*1000)
    print('FPS: ', len(val_dataset)/float(infer_time))
    print('ed1-Acc: ', ed1/len(val_dataset))

    print('C-Acc: ', n_C_correct/len(val_dataset))
    print('woC-Acc: ', n_woC_correct/len(val_dataset))
    print('Acc: ',n_correct, len(val_dataset), acc)
    print('Char-Acc: ',char_acc, char_acc_num,char_acc/1.0/char_acc_num)
    # print 
    return acc, all_predict, confuse_matrix


def visualize_matrix(C):
    plt.rcParams['figure.dpi'] = 500
    
    plt.matshow(C, cmap=plt.cm.Greens) 
    plt.colorbar()
    plt.ylabel('Predicted label')
    plt.xlabel('True label') 
    plt.savefig("filename.png")
    # plt.show()
    return 1

def visualize_matrix2(C):

    df=pd.DataFrame(C,index=alphabets_dict,columns=alphabets_dict)
    sns_plot = sns.heatmap(df,annot=True)
    # plt.savefig("filename.png")
    sns_plot.figure.savefig("output.png")
    # plt.show()
    return 1

def normalize_matrix(C):
    C = C.astype('float')
    columns_sum = np.sum(C, axis=0)
    # print(columns_sum)
    for i in range(C.shape[0]):
        C[i] = C[i]/columns_sum
        # print(C[i])
    # C = C/columns_sum
    # print('C:--------')
    # print(C)
    # print(np.sum(C, axis=0))

    # print(np.sum(C, axis=1))

    return C



if __name__ == '__main__':

    lpr_model = model.LPR_model(1, nclass, imgW=params.imgW, imgH=params.imgH,  K=params.K, isSeqModel=params.isSeqModel, head=params.head, inner=params.inner, isl2Norm=params.isl2Norm).cuda()
    lpr_model.load_state_dict(torch.load(params.model_path, map_location='cuda:0'))

    converter = utils.strLabelConverter(params.alphabet)

    start_time = time.time()
    val_dataset = LPR_LMDB_Dataset(params.image_path)
    val_loader = DataLoader(val_dataset, batch_size=params.val_batchSize, shuffle=False, num_workers=params.workers)
    
    acc, preds, confuse_matrix = val(lpr_model, val_loader)
    confuse_matrix = normalize_matrix(confuse_matrix)
    

    end_time = time.time()
    print(len(val_dataset))
    print('elaspeTime: ',end_time-start_time)
