# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import sys
import time

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

from data_loader import get_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

def main(args):
    # dataloader

    # 

if __name__ == '__main__':
    parser.add_argument(
        '--dataset',
        type=str,
        default='ade20k',
        choices=['coco', 'voc', 'nuswide', 'ade20k', 'recipe1m'])

    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--lr_decay_every', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=1235)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    # Flags
    parser.add_argument('--notensorboard', dest='tensorboard', action='store_false')
    parser.set_defaults(tensorboard=True)
    parser.add_argument('--nodecay_lr', dest='decay_lr', action='store_false')
    parser.set_defaults(decay_lr=True)

    args = parser.parse_args()

    main(args)
