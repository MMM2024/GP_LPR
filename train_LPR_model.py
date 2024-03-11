from __future__ import print_function
from torch.utils.data import DataLoader
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils
import models.LPR_model as model
import torch.nn.init as init

from thop import profile


# import models.LPR_inves_model as model

import re
import params_LPR as params


# from dataset_own import LPRDataset
from dataset_own_lmdb import LPR_LMDB_Dataset

from tensorboardX import SummaryWriter

# custom weights initialization called on lpr_model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def val(lpr_model, val_loader, criterion, iteration, max_i=1000, message='Val_acc'):

    print('Start val')
    for p in lpr_model.parameters():
        p.requires_grad = False
    lpr_model.eval()
    i = 0
    n_correct = 0


    for i_batch, (image, label) in enumerate(val_loader):
        image = image.to(device)
        _, preds,_ = lpr_model(image)

        batch_size = image.size(0)
        cost = 0
        preds_all = preds
        preds = torch.chunk(preds, preds.size(1), 1)

        text = converter.encode_list(label, K=params.K)
        text = text.to(device)
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[:,i]
            cost += criterion(item, gt) / batch_size
        niter = iteration * len(val_loader) + i_batch
        writer.add_scalars('Scalar/Val_loss', {'total_loss': cost.data.item()}, niter)

        _, preds_all = preds_all.max(2)
        sim_preds = converter.decode_list(preds_all.data)
        text_label = label
        for pred, target in zip(sim_preds, text_label):
            pred = pred.replace('-', '')
            if pred == target:
                n_correct += 1

        if (i_batch+1)%params.displayInterval == 0:
            print('[%d/%d][%d/%d] loss: %f' %
                      (iteration, params.niter, i_batch, len(val_loader), cost.data))

        if i_batch == max_i:
            break
    writer.add_text('val-sample', 'epoch: %d'%(iteration), iteration)
    for raw_pred, pred, gt in zip(preds_all, sim_preds, text_label):
        raw_pred = raw_pred.data
        pred = pred.replace('-', '')
        print('raw_pred: %-20s, pred: %-20s, gt: %-20s' % (raw_pred, pred, gt))
        writer.add_text('val-sample', 'raw_pred: %-20s, pred: %-20s, gt: %-20s' % (raw_pred, pred, gt), iteration)

    print(n_correct)
    print(i_batch * params.val_batchSize)
    if max_i*params.val_batchSize < len(val_dataset):
        accuracy = n_correct / (max_i*params.val_batchSize)
    else:
        accuracy = n_correct / len(val_dataset)
    print(message+' accuray: %f' % ( accuracy))
    writer.add_scalars('Scalar/Val_acc', {message: accuracy}, iteration)

    return accuracy

def train(lpr_model, train_loader, criterion, iteration):

    for p in lpr_model.parameters():
        p.requires_grad = True
    lpr_model.train()

    for i_batch, (image, label) in enumerate(train_loader):
        image = image.to(device) # image：tensor[128, 1, 32, 96]，label：tuple128
        text = converter.encode_list(label, K=params.K) # tensor[128, 8]
        text = text.to(device)
        _, preds,_ = lpr_model(image)
        batch_size = image.size(0)

        cost = 0
        # costs = []
        preds = torch.chunk(preds, preds.size(1), 1)
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[:,i]
            # cost_item = criterion(item, gt) / batch_size
            cost_item = criterion(item, gt)
            cost += cost_item
            # costs.append(cost_item)
        # summary writer
        niter = iteration * len(train_loader) + i_batch
        writer.add_scalars('Scalar/Train_loss', {'total_loss': cost.data.item()}, niter)


        lpr_model.zero_grad()
        cost.backward()
        optimizer.step()
        # scheduler.step()
        loss_avg.add(cost)
        # if loss_avg.val() < 0.001:
        #     continue
        if (i_batch+1) % params.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f lr: %f' %
                  (iteration, params.niter, i_batch, len(train_loader), loss_avg.val(), scheduler.get_lr()[0]))
            loss_avg.reset()

def main(lpr_model, train_loader, val_loader, val_loader_2, criterion, optimizer):

    lpr_model = lpr_model.to(device)
    certerion = criterion.to(device)
    Iteration = 0
    best_acc = 0
    while Iteration < params.niter:
        train(lpr_model, train_loader, criterion, Iteration)
        
        accuracy = val(lpr_model, val_loader, criterion, Iteration, max_i=params.valInterval)
        
        for p in lpr_model.parameters():
            p.requires_grad = True
        
        if Iteration %1 == 0:
            torch.save(lpr_model.state_dict(), '{0}/model_{1}_{2}.pth'.format(params.experiment, Iteration, accuracy))


        if accuracy > best_acc:
            torch.save(lpr_model.state_dict(), '{0}/best_model.pth'.format(params.experiment))
            best_acc = accuracy
        
        print("is best accuracy: {0}".format(accuracy > params.best_accuracy))
        scheduler.step()
        Iteration+=1

def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero

if __name__ == '__main__':

    if not os.path.exists(params.experiment):
        os.makedirs(params.experiment)
    manualSeed=1111
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    device = torch.device("cuda:"+str(params.gpu) if torch.cuda.is_available() else "cpu")


    writer = SummaryWriter(comment=params.tb_message)
    writer.add_text('model', 'this model is '+params.tb_message)
    writer.add_text('model_save_path', params.experiment)
    writer.add_text('Train_data', params.train_data)
    writer.add_text('Val_data', params.val_data)


    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')
    # read train set
    dataset = LPR_LMDB_Dataset(params.train_data, isAug=True, isdegrade=True)
    val_dataset = LPR_LMDB_Dataset(params.val_data, isAug=False, isdegrade=False)
    
    train_loader = DataLoader(dataset, batch_size=params.batchSize, shuffle=True, num_workers=params.workers)
    print('train_loader', len(train_loader))
    # shuffle=True, just for time consuming.
    val_loader = DataLoader(val_dataset, batch_size=params.val_batchSize, shuffle=False, num_workers=params.workers)
    val_loader_2 = None

    converter = utils.strLabelConverter(params.alphabet)
    nclass = len(params.alphabet) + 1
    nc = 1

    criterion = torch.nn.CrossEntropyLoss()

    # cnn and rnn
    if params.model_type == 'LPR_model':
        lpr_model = model.LPR_model(nc, nclass, imgW=params.imgW, imgH=params.imgH,  K=params.K).to(device)
    elif params.model_type == 'LPR_SA_model':
        lpr_model = model.LPR_SA_model(nc, nclass, imgW=params.imgW, imgH=params.imgH,  K=params.K, isSeqModel=params.isSeqModel, head=params.head, inner=params.inner, isl2Norm=params.isl2Norm).to(device)
    writer.add_graph(lpr_model, (torch.zeros(1,1,params.imgH,params.imgW).to(device),), False)

    if params.pre_model != '':
        print('loading pretrained model from %s' % params.pre_model)
        lpr_model.load_state_dict(torch.load(params.pre_model, map_location='cuda:0'))
    else:
        lpr_model.apply(weights_init)
        # weights_init(lpr_model)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(lpr_model.parameters(), lr=params.lr,
                               betas=(params.beta1, 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(lpr_model.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(lpr_model.parameters(), lr=params.lr)


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = params.lr_step, gamma = 0.8)
    lpr_model.register_backward_hook(backward_hook)
    main(lpr_model, train_loader, val_loader, val_loader_2, criterion, optimizer)

    writer.close()
