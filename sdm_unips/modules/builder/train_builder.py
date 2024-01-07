"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""


import torch
import torch.nn.functional as F
import numpy as np
from modules.model import model_control as model
from modules.model import decompose_tensors
from tqdm import tqdm
from modules.model.model_utils import *
from modules.utils import compute_mae
import cv2
import glob
from torch.utils.tensorboard import SummaryWriter

class builder():
    def __init__(self, args, device):
   
        self.device = device
        self.args = args    
        self.test_epoch = args.test_epoch
        self.save_model_epoch = args.save_model_epoch
        self.train_epoch = args.train_epoch           
        self.mseloss = torch.nn.MSELoss(reduction='sum')
        """Load pretrained model (normal or brdf)"""

        if 'brdf' in args.target:
            model_dir = f'{args.checkpoint}/brdf'
            self.net_brdf = model.Net(args.pixel_samples, 'brdf', device).to(self.device)
            self.net_brdf = torch.nn.DataParallel(self.net_brdf)
            self.net_brdf = self.load_models(self.net_brdf, model_dir)
            self.net_brdf.module.no_grad()

        # required_optimization = []
        # for param in self.net_brdf.parameters():
        #     if param.requires_grad is True:
        #         required_optimization.append(param)
        self.optimizer_brdf = torch.optim.Adam(self.net_brdf.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.summary=SummaryWriter(log_dir=f'results/log')
        print('')

        print(f"canonical resolution: {self.args.canonical_resolution} x {self.args.canonical_resolution}  ")
        print(f"pixel samples: {self.args.pixel_samples}\n") 

    def separate_batch(self, batch):
        I = batch[0].to(self.device) # [B, 3, H, W]
        N = batch[1].to(self.device) # [B, 1, H, W]
        M = batch[2].to(self.device) # [B, 1, H, W]
        nImgArray = batch[3]
        roi = batch[4]
        L = batch[5].to(self.device) # [B, 3, H, W]
        albedo = batch[6].to(self.device) # [B, 3, H, W]
        roughness = batch[7].to(self.device)
        metallic = batch[8].to(self.device)
        return I, N, M, nImgArray, roi, L, albedo, roughness, metallic

    def load_models(self, model, dirpath):
        pytmodel = "".join(glob.glob(f'{dirpath}/*.pytmodel'))
        model = loadmodel(model, pytmodel, strict=False)
        return model

    def run(self, 
            canonical_resolution = None,
            data = None,
            max_image_resolution = None,
            ):
        
        data.max_image_resolution = max_image_resolution
        test_data_loader = torch.utils.data.DataLoader(data, batch_size = 1, shuffle=False, num_workers=0, pin_memory=True)
        self.net_brdf.train()
        train_idx = -1
        for idx in tqdm(range(self.train_epoch)):

            for batch_test in test_data_loader:    
                train_idx+=1                  
                I, N, M, nImgArray, roi, L, albedo, roughness, metallic = self.separate_batch(batch_test)
                roi = roi[0].numpy()
                h_ = roi[0]
                w_ = roi[1]
                r_s = roi[2]
                r_e = roi[3]
                c_s = roi[4]
                c_e = roi[5]
                B, C, H, W, Nimg = I.shape
                if self.args.scalable:    
                    patch_size = 64               
                    patches_I = decompose_tensors.divide_tensor_spatial(I.permute(0,4,1,2,3).reshape(-1, C, H, W), block_size=patch_size, method='tile_stride')
                    patches_I = patches_I.reshape(B, Nimg, -1, C, patch_size, patch_size).permute(0, 2, 3, 4, 5, 1)
                    sliding_blocks = patches_I.shape[1]
                    patches_M = decompose_tensors.divide_tensor_spatial(M, block_size=patch_size, method='tile_stride')
                    
                    patches_nml = []
                    patches_base = []
                    patches_rough = []
                    patches_metal = []

                    for k in range(sliding_blocks):
                        print(f"Recovering {self.args.target} map(s): {k+1} / {sliding_blocks}")
                        if torch.sum(patches_M[:, k, :, :, :]) > 0:
                            pI = patches_I[:, k, :, :, :,:]
                            pI = F.interpolate(pI.permute(0,4,1,2,3).reshape(-1, pI.shape[1],pI.shape[2],pI.shape[3]), size=(patch_size, patch_size), mode='bilinear', align_corners=True).reshape(B, Nimg, C, patch_size, patch_size).permute(0,2,3,4,1)
                            pM = F.interpolate(patches_M[:, k, :, :, :], size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                            nout = torch.zeros((B, 3, patch_size, patch_size))
                            bout = torch.zeros((B, 3, patch_size, patch_size))
                            rout = torch.zeros((B, 1, patch_size, patch_size))
                            mout = torch.zeros((B, 1, patch_size, patch_size))
    
                            if 'brdf' in self.args.target:
                                nout, bout, rout, mout  = self.net_brdf(pI, pM, L, nImgArray.reshape(-1,1), decoder_resolution = patch_size * torch.ones(pI.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(pI.shape[0],1))
                                nout = (F.interpolate(nout, size=(patch_size, patch_size), mode='bilinear', align_corners=True) * pM)

                                bout = F.interpolate(bout, size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                                rout = F.interpolate(rout, size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                                mout = F.interpolate(mout, size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                            patches_nml.append(nout)
                            patches_base.append(bout)
                            patches_rough.append(rout)
                            patches_metal.append(mout)
                        else:
                            patches_nml.append(torch.zeros((B, 3, patch_size, patch_size)))   
                            patches_base.append(torch.zeros((B, 3, patch_size, patch_size)))     
                            patches_rough.append(torch.zeros((B, 1, patch_size, patch_size)))          
                            patches_metal.append(torch.zeros((B, 1, patch_size, patch_size)))           
                    patches_nml = torch.stack(patches_nml, dim=1)
                    patches_base = torch.stack(patches_base, dim=1)
                    patches_rough = torch.stack(patches_rough, dim=1)
                    patches_metal = torch.stack(patches_metal, dim=1)
                    merged_tensor_nml = decompose_tensors.merge_tensor_spatial(patches_nml.permute(1,0,2,3,4), method='tile_stride')
                    merged_tensor_base = decompose_tensors.merge_tensor_spatial(patches_base.permute(1,0,2,3,4), method='tile_stride')
                    merged_tensor_rough = decompose_tensors.merge_tensor_spatial(patches_rough.permute(1,0,2,3,4), method='tile_stride')
                    merged_tensor_metal = decompose_tensors.merge_tensor_spatial(patches_metal.permute(1,0,2,3,4), method='tile_stride')
                    nml = merged_tensor_nml
                    base = merged_tensor_base
                    rough = merged_tensor_rough
                    metal = merged_tensor_metal


                                  
                    nml = F.interpolate(nml, size=(r_e-r_s, c_e-c_s), mode='bicubic', align_corners=True).squeeze().permute(1,2,0)
                    mask = (torch.abs(1 - torch.sqrt(torch.sum(nml * nml, dim=2))) < 0.5).float()
                    nml = F.normalize(nml, dim=2)
                    nml = nml * mask[:, :, None]
                    nout = torch.zeros((h_, w_, 3)).to(self.device)
                    nout[r_s:r_e, c_s:c_e,:] = nml
                    # nimg = nout.cpu().detach().numpy()
                    # cv2.imwrite(f'{data.data.data_workspace}/normal.png', 255*(0.5 * (1+nimg[:,:,::-1])))  


                    base = F.interpolate(base, size=(r_e-r_s, c_e-c_s), mode='bicubic', align_corners=True).squeeze().permute(1,2,0)
                    bout = torch.zeros((h_, w_, 3)).to(self.device)
                    bout[r_s:r_e, c_s:c_e,:] = base

                    rough = F.interpolate(rough, size=(r_e-r_s, c_e-c_s), mode='bicubic', align_corners=True).squeeze()
                    rout = torch.zeros((h_, w_)).to(self.device)
                    rout[r_s:r_e, c_s:c_e] = rough

                    metal = F.interpolate(metal, size=(r_e-r_s,c_e-c_s), mode='bicubic', align_corners=True).squeeze()
                    mout = torch.zeros((h_, w_)).to(self.device)
                    mout[r_s:r_e, c_s:c_e] = metal


                    n_true = N.permute(0,2,3,1).squeeze()  
                    mask = (torch.abs(1 - torch.sqrt(torch.sum(n_true * n_true, axis=2))) < 0.5).float()      
                    num = torch.sum(mask)
                    loss_n = self.mseloss(nout, N.permute(0,2,3,1).squeeze())/torch.sum(mask)
                    loss_b = self.mseloss(bout, albedo.permute(0,2,3,1).squeeze())/torch.sum(mask)
                    loss_r = self.mseloss(rout, roughness)/torch.sum(mask)
                    loss_m = self.mseloss(mout, metallic)/torch.sum(mask)

                    loss = (loss_n + loss_b + loss_r + loss_m)/4
                    print(f"loss: {loss.item():.3f}","loss_n: ",loss_n.item(),"loss_b: ",loss_b.item(),"loss_r: ",loss_r.item(),"loss_m: ",loss_m.item())
                    
                    self.summary.add_scalar('loss', loss.item(), train_idx)
                    self.summary.add_scalar('loss_n', loss_n.item(), train_idx)
                    self.summary.add_scalar('loss_b', loss_b.item(), train_idx)
                    self.summary.add_scalar('loss_r', loss_r.item(), train_idx)
                    self.summary.add_scalar('loss_m', loss_m.item(), train_idx)
                    loss.backward()
                    self.optimizer_brdf.step()
                    # self.net_brdf.zero_grad()
                    self.optimizer_brdf.zero_grad()


                    nout = nout.cpu().detach().numpy()
                    n_true = N.permute(0,2,3,1).squeeze().cpu().numpy()
                    mask = np.float32(np.abs(1 - np.sqrt(np.sum(n_true * n_true, axis=2))) < 0.5)
                    mae, emap = compute_mae.compute_mae_np(nout, n_true, mask = mask)
                    print(f"Mean Angular Error (MAE) is {mae:.3f}\n")           
                    self.summary.add_scalar('MAE', mae, train_idx)             
                    emap = emap.squeeze()
                    thresh = 90
                    emap[emap>=thresh] = thresh
                    emap = emap/thresh
                    cv2.imwrite(f'{data.data.data_workspace}/error.png', 255*emap)   

                if train_idx % self.test_epoch == 0:
                    with torch.no_grad():  
                        self.net_brdf.eval()
                        if self.args.scalable:    
                            patch_size = 64               
                            patches_I = decompose_tensors.divide_tensor_spatial(I.permute(0,4,1,2,3).reshape(-1, C, H, W), block_size=patch_size, method='tile_stride')
                            patches_I = patches_I.reshape(B, Nimg, -1, C, patch_size, patch_size).permute(0, 2, 3, 4, 5, 1)
                            sliding_blocks = patches_I.shape[1]
                            patches_M = decompose_tensors.divide_tensor_spatial(M, block_size=patch_size, method='tile_stride')
                            
                            patches_nml = []
                            patches_base = []
                            patches_rough = []
                            patches_metal = []

                            for k in range(sliding_blocks):
                                print(f"Recovering {self.args.target} map(s): {k+1} / {sliding_blocks}")
                                if torch.sum(patches_M[:, k, :, :, :]) > 0:
                                    pI = patches_I[:, k, :, :, :,:]
                                    pI = F.interpolate(pI.permute(0,4,1,2,3).reshape(-1, pI.shape[1],pI.shape[2],pI.shape[3]), size=(patch_size, patch_size), mode='bilinear', align_corners=True).reshape(B, Nimg, C, patch_size, patch_size).permute(0,2,3,4,1)
                                    pM = F.interpolate(patches_M[:, k, :, :, :], size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                                    nout = torch.zeros((B, 3, patch_size, patch_size))
                                    bout = torch.zeros((B, 3, patch_size, patch_size))
                                    rout = torch.zeros((B, 1, patch_size, patch_size))
                                    mout = torch.zeros((B, 1, patch_size, patch_size))
            
                                    if 'brdf' in self.args.target:
                                        nout, bout, rout, mout  = self.net_brdf(pI, pM, L, nImgArray.reshape(-1,1), decoder_resolution = patch_size * torch.ones(pI.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(pI.shape[0],1))
                                        nout = (F.interpolate(nout, size=(patch_size, patch_size), mode='bilinear', align_corners=True) * pM).cpu()

                                        bout = F.interpolate(bout, size=(patch_size, patch_size), mode='bilinear', align_corners=True).cpu()
                                        rout = F.interpolate(rout, size=(patch_size, patch_size), mode='bilinear', align_corners=True).cpu()
                                        mout = F.interpolate(mout, size=(patch_size, patch_size), mode='bilinear', align_corners=True).cpu()
                                    patches_nml.append(nout)
                                    patches_base.append(bout)
                                    patches_rough.append(rout)
                                    patches_metal.append(mout)
                                else:
                                    patches_nml.append(torch.zeros((B, 3, patch_size, patch_size)))   
                                    patches_base.append(torch.zeros((B, 3, patch_size, patch_size)))     
                                    patches_rough.append(torch.zeros((B, 1, patch_size, patch_size)))          
                                    patches_metal.append(torch.zeros((B, 1, patch_size, patch_size)))           
                            patches_nml = torch.stack(patches_nml, dim=1)
                            patches_base = torch.stack(patches_base, dim=1)
                            patches_rough = torch.stack(patches_rough, dim=1)
                            patches_metal = torch.stack(patches_metal, dim=1)
                            merged_tensor_nml = decompose_tensors.merge_tensor_spatial(patches_nml.permute(1,0,2,3,4), method='tile_stride')
                            merged_tensor_base = decompose_tensors.merge_tensor_spatial(patches_base.permute(1,0,2,3,4), method='tile_stride')
                            merged_tensor_rough = decompose_tensors.merge_tensor_spatial(patches_rough.permute(1,0,2,3,4), method='tile_stride')
                            merged_tensor_metal = decompose_tensors.merge_tensor_spatial(patches_metal.permute(1,0,2,3,4), method='tile_stride')
                            nml = merged_tensor_nml.squeeze().permute(1,2,0)
                            base = merged_tensor_base.squeeze().permute(1,2,0)
                            rough = merged_tensor_rough.squeeze()
                            metal = merged_tensor_metal.squeeze()
                        else:
                            print(f"Recovering {self.args.target} map(s) 1 / 1")
                            
                            if 'brdf' in self.args.target:
                                nout, bout, rout, mout  = self.net_brdf(I, M, nImgArray.reshape(-1,1), decoder_resolution = data.data.h * torch.ones(I.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(I.shape[0],1))
                                nml = (nout * M).squeeze().permute(1,2,0).cpu().detach()
                                base = (bout * M).squeeze().permute(1,2,0).cpu().detach()
                                rough = (rout * M).squeeze().cpu().detach()
                                metal = (mout * M).squeeze().cpu().detach()
                                del bout, rout, mout
                        
                        # save normal of original resolution
                        if 'brdf' in self.args.target:
                            nml = nml.cpu().numpy()                
                            nml = cv2.resize(nml, dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_CUBIC)
                            mask = np.float32(np.abs(1 - np.sqrt(np.sum(nml * nml, axis=2))) < 0.5)
                            nml = np.divide(nml, np.linalg.norm(nml, axis=2, keepdims=True) + 1.0e-12)
                            nml = nml * mask[:, :, np.newaxis]
                            nout = np.zeros((h_, w_, 3), np.float32)
                            nout[r_s:r_e, c_s:c_e,:] = nml

                            if torch.sum(N) > 0:
                                n_true = N.permute(0,2,3,1).squeeze().cpu().numpy()
                                mask = np.float32(np.abs(1 - np.sqrt(np.sum(n_true * n_true, axis=2))) < 0.5)
                                mae, emap = compute_mae.compute_mae_np(nout, n_true, mask = mask)
                                print(f"Mean Angular Error (MAE) is {mae:.3f}\n")                        
                                emap = emap.squeeze()
                                thresh = 90
                                emap[emap>=thresh] = thresh
                                emap = emap/thresh
                                cv2.imwrite(f'{data.data.data_workspace}/error.png', 255*emap)     
                            
                            cv2.imwrite(f'{data.data.data_workspace}/{idx}_normal.png', 255*(0.5 * (1+nout[:,:,::-1])))                                      

                        if 'brdf' in self.args.target:
                            base = cv2.resize(base.cpu().numpy(), dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_CUBIC)
                            rough = cv2.resize(rough.cpu().numpy(), dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_CUBIC)
                            metal = cv2.resize(metal.cpu().numpy(), dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_CUBIC)

                            bout = np.zeros((h_, w_, 3), np.float32)
                            bout[r_s:r_e, c_s:c_e,:] = base
                            cv2.imwrite(f'{data.data.data_workspace}/{idx}_baseColor.png', 255*bout[:,:,::-1])

                            rout = np.zeros((h_, w_), np.float32)
                            rout[r_s:r_e, c_s:c_e] = rough
                            cv2.imwrite(f'{data.data.data_workspace}/{idx}_roughness.png', 255*rout[:,:])

                            mout = np.zeros((h_, w_), np.float32)
                            mout[r_s:r_e, c_s:c_e] = metal
                            cv2.imwrite(f'{data.data.data_workspace}/{idx}_metallic.png', 255*mout[:,:])
                        self.net_brdf.train()




