#! /usr/bin/env python

'''

This file is written based on the Trainer.py. It is written for testing
Author: Hongtao Wu
Contact: hwu67@jhu.edu
Aug 14, 2019

'''


import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.autograd import Variable

class ModelNetTester(object):

    def __init__(self, model, test_loader, loss_fn, model_name, weight_path, weight_name, num_views=12):

        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.num_views = num_views

        self.model.cuda()
        
        self.model.load(weight_path, weight_name)


    def test(self):
        all_correct_points = 0
        all_points = 0

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for _, data in enumerate(self.test_loader, 0):

            N,V,C,H,W = data[1].size()
            in_data = Variable(data[1]).view(-1,C,H,W).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target
            
            model_name = data[2]
            print(model_name)

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print ('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.test_loader)

        print ('test mean class acc. : ', val_mean_class_acc)
        print ('test overall acc. : ', val_overall_acc)
        print ('test loss : ', loss)

        return loss, val_overall_acc, val_mean_class_acc

