import torch
import argparse
from torch.autograd import Variable
import os
import json
import random
import numpy as np
import time

from resnet import resnet50
from MetaAge_model import *
from loss import *
from utils import *
from dataset import load_data

dtype = torch.float32
USE_GPU = True
EPOCH = 60
BATCH_SIZE = 64
print_every = int(10 / BATCH_SIZE * 64)
Load_model = False
NAME = 'MORPHII'

NET_LR = 1e-4
FC_LR = 1e-4
OPTIMIZER = 'adam'
LR_DECAY_EPOCH = [] if OPTIMIZER == 'adam' else [15, 30]
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
DECAY_RATE = 0.1
RANDOM_SEED = 6383
print("RANDOM SEED:",RANDOM_SEED)

print("hinge :%.4f, DELTA:%.4f"%(HINGE_LAMBDA, DELTA))

START_EPOCH = 0

CROSS_VAL_I = 0

loader_train = None
loader_val = None

def set_device():
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('using device:', device)
    return device

def accuracy_mae(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    if True:
        maxk = max(topk)
        batch_size = target.size(0)

        target_int = target.to(dtype=torch.long)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target_int.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc_k = correct_k.mul_(100.0 / batch_size)
            acc_k = acc_k.cpu().data.numpy()[0]
            res.append(acc_k)

        softmax_layer = nn.Softmax(dim=1)
        preb = softmax_layer(output)
        preb_data = preb.cpu().data.numpy()
        target_data = target.cpu().data.numpy()
        label_arr = np.array(range(101))
        estimate_ages = np.sum(preb_data * label_arr, axis=1)
        age_dis = abs(estimate_ages - target_data)
        mae = sum(age_dis) * 1.0 / len(target_data)   
        res.append(mae)

        return res

def adjust_learning_rate(optimizer, decay_rate=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print('learning rate: %.2e' % param_group['lr'])

def train(model, optimizer, criterion, device, epochs=1, start=0):
    global loader_train, loader_val
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    bestMAE_ever = 100

    if not os.path.isdir(NAME + '_save'):
        os.mkdir(NAME + '_save')

    val_acc = 0
    for e in range(start, epochs):
        best_optimizer = optimizer
        best_model = model
        best_acc = 0

        losses = AverageMeter()
        softmax_losses = AverageMeter()
        hinge_losses = AverageMeter()
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        mae = AverageMeter()

        if e in LR_DECAY_EPOCH:
            adjust_learning_rate(optimizer, decay_rate=DECAY_RATE)

        end_time = time.time()
        for t, (x224, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x224 = x224.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad()

            output = model(x224)

            loss, softmax_loss, hinge_loss = criterion(output, y)

            loss.backward()

            optimizer.step()

            prec1, prec5, batch_mae = accuracy_mae(output, y, topk=(1, 5))
            losses.update(loss.item())
            if HINGE_LOSS_ENABLE:
              softmax_losses.update(softmax_loss.item())
              hinge_losses.update(hinge_loss.item())
            top1.update(prec1)
            top5.update(prec5)
            mae.update(batch_mae)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if t % print_every == 0:
                print('Train: [%d/%d]\t'
                      'Time %.3f (%.3f)\t'
                      'Loss %.4f (%.4f)\t'
                      'softmax Loss %.4f (%.4f)\t'
                      'hinge Loss %.4f (%.4f)\t'
                      'Prec@1 %.3f (%.3f)\t'
                      'Prec@5 %.3f (%.3f)\t'
                      'MAE %.3f (%.3f)' % (t, len(loader_train), batch_time.val, batch_time.avg,losses.val, losses.avg, softmax_losses.val, softmax_losses.avg, hinge_losses.val, hinge_losses.avg, top1.val, top1.avg, top5.val, top5.avg, mae.val, mae.avg))

        MAE_val = test_epoch(model, criterion, loader_val, device, e, epochs)

        if MAE_val < bestMAE_ever:
            bestMAE_ever = MAE_val
            save_model_optimizer_history(model, optimizer, filepath=NAME + '_save' + '/epoch%d_MAE_%.3f' % (e, MAE_val),
                                    device=device)

    print("best MAE:", bestMAE_ever)


def test_epoch(model, criterion, testloader, device, epoch, end_epoch, verbo=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mae = AverageMeter()

    model.eval()
    total = 0
    correct = 0
    all_mae = 0
    end_time = time.time()
    for batch_idx, (x224, targets) in enumerate(testloader):
        x224 = x224.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        targets = targets.to(device=device, dtype=torch.long)
        x224, targets = Variable(x224), Variable(targets)

        output = model(x224)
        output = output.to(device=device)

        loss, _, _ = criterion(output, targets)

        _, predicted = output.max(1)
        total += targets.size(0)
        target_int = targets.to(device=device, dtype=torch.long)
        correct += predicted.eq(target_int).sum().cpu().data.numpy()

        prec1, prec5, batch_mae = accuracy_mae(output, targets, topk=(1, 5))
        losses.update(loss.cpu().data.numpy())
        top1.update(prec1)
        top5.update(prec5)
        mae.update(batch_mae)
        all_mae = all_mae + batch_mae * targets.size(0)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_idx % 20 == 0 and verbo == True:
            print('Test: [%d/%d]\t'
                  'Time %.3f (%.3f)\t'
                  
                  'Loss %.4f (%.4f)\t'
                  'Prec@1 %.3f (%.3f)\t'
                  'Prec@5 %.3f (%.3f)\t'
                  'MAE %.3f (%.3f)' % (batch_idx, len(testloader),
                                       batch_time.val, batch_time.avg,
                                       losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg,
                                       mae.val, mae.avg))

    acc = 100. * correct / total
    print('Test: [%d/%d] Acc %.3f MAE: %.3f Prec@5: %.3f' % (epoch, end_epoch, acc, all_mae * 1.0 / total, top5.val))
    return all_mae * 1.0 / total

def is_fc(para_name):
    split_name = para_name.split('.')
    if len(split_name) < 3:
        return False
    if split_name[-3] == 'classifier':
        return True
    else:
        return False

def net_lr(model, fc_lr, lr):
    params = []
    for keys, param_value in model.named_parameters():
        if (is_fc(keys)):
            params += [{'params': [param_value], 'lr': fc_lr}]
        else:
            params += [{'params': [param_value], 'lr': lr}]
    return params


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_main(pretrained_vgg_path, pretrained_resnet_path, list_root, pic_root_dir):
    global loader_train, loader_val
    loader_train, loader_val = load_data(BATCH_SIZE, list_root, pic_root_dir, RANDOM_SEED)

    device = set_device()
    setup_seed(RANDOM_SEED)

    vgg_model = load_pretrained_model(pretrained_vgg_path)
    face_feature_model = resnet50(pretrained_resnet_path)
    model = MetaAge(vgg_model=vgg_model, face_feature_model=face_feature_model, device=device)
    model = nn.DataParallel(model)

    criterion2 = loss_func(device)

    params = net_lr(model, FC_LR, NET_LR)

    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999), weight_decay=0, eps=1e-08)
    else:
        optimizer = torch.optim.SGD(params, momentum=MOMENTUM, nesterov=True,
                                    weight_decay=WEIGHT_DECAY)

    print(model)
    start_epoch = 0
    if Load_model:
        start_epoch = 25
        filepath = 'load_model_path'
        model = load_model(model, filepath, device=device)
        model = model.to(device=device)
        optimizer = load_optimizer(optimizer, filepath, device=device)

    train(model, optimizer, criterion2, device=device, epochs=EPOCH, start=start_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch age estimation')
    parser.add_argument('--pretrained_vgg_path', type=str, help='path of pretrained model')
    parser.add_argument('--pretrained_resnet_path', type=str, help='path of pretrained resnet model')
    parser.add_argument('--list_root', type=str, help='path of data list')
    parser.add_argument('--pic_root_dir', type=str, help='path of picture directory')
    args = parser.parse_args()

    train_main(args.pretrained_vgg_path, args.pretrained_resnet_path, args.list_root, args.pic_root_dir)
