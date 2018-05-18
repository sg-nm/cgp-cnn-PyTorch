#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import random
from skimage.measure import compare_psnr
import os

from cnn_model import CGP2CNN
from my_data_loader import get_train_valid_loader, get_test_loader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)



# __init__: load dataset
# __call__: training the CNN defined by CGP list
class CNN_train():
    def __init__(self, dataset_name, validation=True, verbose=True, imgSize=32, batchsize=128):
        # dataset_name: name of data set ('bsds'(color) or 'bsds_gray')
        # validation: [True]  model train/validation mode
        #             [False] model test mode for final evaluation of the evolved model
        #                     (raining data : all training data, test data : all test data)
        # verbose: flag of display
        self.verbose = verbose
        self.imgSize = imgSize
        self.validation = validation
        self.batchsize = batchsize
        self.dataset_name = dataset_name

        # load dataset
        if dataset_name == 'cifar10' or dataset_name == 'mnist':
            if dataset_name == 'cifar10':
                self.n_class = 10
                self.channel = 3
                if self.validation:
                    self.dataloader, self.test_dataloader = get_train_valid_loader(data_dir='./', batch_size=self.batchsize, augment=True, random_seed=2018, num_workers=1, pin_memory=True)
                    # self.dataloader, self.test_dataloader = loaders[0], loaders[1]
                else:
                    train_dataset = dset.CIFAR10(root='./', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.Scale(self.imgSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ]))
                    test_dataset = dset.CIFAR10(root='./', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.Scale(self.imgSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ]))
                    self.dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=int(2))
                    self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize, shuffle=True, num_workers=int(2))
            print('train num    ', len(self.dataloader.dataset))
            # print('test num     ', len(self.test_dataloader.dataset))
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

    def __call__(self, cgp, gpuID, epoch_num=200, out_model='mymodel.model'):
        if self.verbose:
            print('GPUID     :', gpuID)
            print('epoch_num :', epoch_num)
            print('batch_size:', self.batchsize)
        
        # model
        torch.backends.cudnn.benchmark = True
        model = CGP2CNN(cgp, self.channel, self.n_class, self.imgSize)
        init_weights(model, 'kaiming')
        model.cuda(gpuID)
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        criterion.cuda(gpuID)
        optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=0.0005)
        input = torch.FloatTensor(self.batchsize, self.channel, self.imgSize, self.imgSize)
        input = input.cuda(gpuID)
        label = torch.LongTensor(self.batchsize)
        label = label.cuda(gpuID)

        # Train loop
        for epoch in range(1, epoch_num+1):
            start_time = time.time()
            if self.verbose:
                print('epoch', epoch)
            train_loss = 0
            total = 0
            correct = 0
            ite = 0
            for module in model.children():
                module.train(True)
            for _, (data, target) in enumerate(self.dataloader):
                if self.dataset_name == 'mnist':
                    data = data[:,0:1,:,:] # for gray scale images
                data = data.cuda(gpuID)
                target = target.cuda(gpuID)
                input.resize_as_(data).copy_(data)
                input_ = Variable(input)
                label.resize_as_(target).copy_(target)
                label_ = Variable(label)
                optimizer.zero_grad()
                try:
                    output = model(input_, None)
                except:
                    import traceback
                    traceback.print_exc()
                    return 0.
                loss = criterion(output, label_)
                train_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output.data, 1)
                total += label_.size(0)
                correct += predicted.eq(label_.data).cpu().sum()
                ite += 1
            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('Train set : Average Acc : {:.4f}'.format(correct/total))
            print('time ', time.time()-start_time)
            if self.validation:
                if epoch == 30:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch == epoch_num:
                    for module in model.children():
                        module.train(False)
                    t_loss = self.__test_per_std(model, criterion, gpuID, input, label)
            else:
                if epoch == 5:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 10
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch % 10 == 0:
                    for module in model.children():
                        module.train(False)
                    t_loss = self.__test_per_std(model, criterion, gpuID, input, label)
                if epoch == 250:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch == 375:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
        # save the model
        torch.save(model.state_dict(), './model_%d.pth' % int(gpuID))
        return t_loss

    # For validation/test
    def __test_per_std(self, model, criterion, gpuID, input, label):
        test_loss = 0
        total = 0
        correct = 0
        ite = 0
        for _, (data, target) in enumerate(self.test_dataloader):
            if self.dataset_name == 'mnsit':
                data = data[:,0:1,:,:]
            data = data.cuda(gpuID)
            target = target.cuda(gpuID)
            input.resize_as_(data).copy_(data)
            input_ = Variable(input)
            label.resize_as_(target).copy_(target)
            label_ = Variable(label)
            try:
                output = model(input_, None)
            except:
                import traceback
                traceback.print_exc()
                return 0.
            loss = criterion(output, label_)
            test_loss += loss.data[0]
            _, predicted = torch.max(output.data, 1)
            total += label_.size(0)
            correct += predicted.eq(label_.data).cpu().sum()
            ite += 1
        print('Test set : Average loss: {:.4f}'.format(test_loss))
        print('Test set : (%d/%d)' % (correct, total))
        print('Test set : Average Acc : {:.4f}'.format(correct/total))

        return (correct/total)
