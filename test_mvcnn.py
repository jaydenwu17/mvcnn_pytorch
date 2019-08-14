#! /usr/bin/env python

'''

This code is for testing the MVCNN.
Author: Hongtao Wu
Contact: hwu67@jhu.edu
Aug 14, 2019

'''

import numpy as np
import torch
import torch.nn as nn 
import argparse

from tools.Tester import ModelNetTester
from tools.ImgDataset import MultiviewImgDataset
from models.MVCNN import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-test_path", type=str, default="/disk1/spirit-dictionary/baseline/deeperlook/chair_imagine_test/synthetic/upright/airplane/test")
parser.add_argument("-weight_path", type=str, default="/home/hongtao/src/mvcnn_pytorch/MVCNN_stage_2")
parser.add_argument("-weight_name", type=str, default=None)


if __name__ == '__main__':
    args = parser.parse_args()
    
    cnet = SVCNN(args.name, nclasses=40, pretraining=False, cnn_name=args.cnn_name)
    cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)

    del cnet

    test_dataset = MultiviewImgDataset(args.test_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("Model Number: {}".format(len(test_loader)))
    tester = ModelNetTester(cnet_2, test_loader, nn.CrossEntropyLoss(), 'mvcnn', args.weight_path, args.weight_name)

    tester.test()
    print("Finished!")
