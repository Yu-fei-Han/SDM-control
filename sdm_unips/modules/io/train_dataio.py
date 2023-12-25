"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import glob
import torch.utils.data as data
from .dataloader import realdata_train
import numpy as np

class dataio(data.Dataset):
    def __init__(self, mode, args):
        self.mode = mode
        data_root = args.train_dir
        extension = args.train_ext
        self.numberOfImageBuffer = args.max_image_num
        self.train_image_prefix= args.train_image_prefix
        self.train_light_prefix= args.train_light_prefix
        self.mask_margin = args.mask_margin
        self.outdir = args.session_name
        self.data_root = data_root
        self.extension = extension
        self.data_name = []
        self.set_id = []
        self.valid = []
        self.sample_id = []
        self.dataCount = 0
        self.dataLength = -1
        self.max_image_resolution = None
        
        print('Exploring %s' % (data_root))
        objlist = glob.glob(f"{data_root}/*{extension}")
        objlist = sorted(objlist)
        self.objlist = objlist
        print(f"Found {len(self.objlist)} objects!\n")
        self.data = realdata_train.dataloader(self.numberOfImageBuffer, mask_margin=self.mask_margin, outdir=self.outdir)

    def __getitem__(self, index_):

        objid = index_
        objdir = self.objlist[objid]
        self.data.load(objdir, image_prefix = self.train_image_prefix, light_prefix = self.train_light_prefix, max_image_resolution = self.max_image_resolution)
        img = self.data.I.transpose(2,0,1,3) # c, h, w, N
        numberOfImages = self.data.I.shape[3]           
        nml = self.data.N.transpose(2,0,1) # 3, h, w
        mask = np.transpose(self.data.mask, (2,0,1)) # 1, h, w
        roi = self.data.roi
        light = self.data.L.transpose(2,0,1,3) # c, h, w, N
        albedo = self.data.a.transpose(2,0,1) # c, h, w
        roughness = self.data.r.transpose(2,0,1)
        mettalic = self.data.m.transpose(2,0,1)

        return img, nml, mask, numberOfImages, roi, light, albedo, roughness, mettalic

    def __len__(self):
        return len(self.objlist)
