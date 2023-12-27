"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""


import torch
import torch.nn.functional as F
import numpy as np
from modules.model import model_control, decompose_tensors
from modules.model.model_utils import *
from modules.utils import compute_mae
import cv2
import glob
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 

class builder():
    def __init__(self, args, device):
        self.train_epoch = args.train_epoch
        self.test_interval = 20
        self.device = device
        self.args = args       
        self.mseloss = torch.nn.MSELoss(reduction='sum')        
        """Load pretrained model (normal or brdf)"""
        # if 'normal' in args.target:
        #     model_dir = f'{args.checkpoint}/normal'
        #     self.net_nml = model_control.Net(args.pixel_samples, 'normal', device).to(self.device)
        #     self.net_nml = torch.nn.DataParallel(self.net_nml)
        #     self.net_nml = self.load_models(self.net_nml, model_dir)
        #     self.net_nml.module.no_grad()
        if 'brdf' in args.target:
            model_dir = f'{args.checkpoint}/brdf'
            self.net_brdf = model_control.Net(args.pixel_samples, 'brdf', device).to(self.device)
            self.net_brdf = self.load_models(self.net_brdf, model_dir)
            if not args.fine_tune:
                print("Fine-tuning the network...")
                model_dir = f'{args.checkpoint}/fine_tune'
                self.net_brdf = self.load_models(self.net_brdf, model_dir)
            else:
                self.net_brdf.no_grad()
            self.net_brdf = torch.nn.DataParallel(self.net_brdf)
        print('')
        print(f"canonical resolution: {self.args.canonical_resolution} x {self.args.canonical_resolution}  ")
        print(f"pixel samples: {self.args.pixel_samples}\n") 
        self.writer = SummaryWriter(f'{args.session_name}/logs')

        finetune_params = []
        for name, param in self.net_brdf.named_parameters():
            if param.requires_grad:
                finetune_params.append(param)

        self.optimizer = torch.optim.Adam(finetune_params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

    def separate_batch(self, batch):
        I = batch[0].to(self.device) # [B, 3, H, W]
        N = batch[1].to(self.device) # [B, 1, H, W]
        M = batch[2].to(self.device) # [B, 1, H, W]
        nImgArray = batch[3]
        roi = batch[4]
        L = batch[5].to(self.device)
        a = batch[6].to(self.device)
        r = batch[7].to(self.device)
        m = batch[8].to(self.device)
        return I, N, M, nImgArray, roi, L, a, r, m

    def load_models(self, model, dirpath):
        pytmodel = "".join(glob.glob(f'{dirpath}/*.pytmodel'))
        model = loadmodel(model, pytmodel, strict=False)
        return model

    def save_models(self, model, dirpath):
        torch.save(model.state_dict(), f'{dirpath}/{self.train_epoch:04d}.pytmodel')

    def run(self, 
            canonical_resolution = None,
            data = None,
            max_image_resolution = None,
            ):
        
        data.max_image_resolution = max_image_resolution
        train_data_loader = torch.utils.data.DataLoader(data, batch_size = 1, shuffle=False, num_workers=0, pin_memory=True)
        test_epoch = 0
        self.net_brdf.train()
        for iter in tqdm(range(self.train_epoch)):

            for batch_train in train_data_loader:                      
                I, N, M, nImgArray, roi, L, a, r, m = self.separate_batch(batch_train)
                roi = roi[0].numpy()
                h_ = roi[0]
                w_ = roi[1]
                r_s = roi[2]
                r_e = roi[3]
                c_s = roi[4]
                c_e = roi[5]
                B, C, H, W, Nimg = I.shape
                N = N.to(self.device)
                a = a.to(self.device)
                r = r.to(self.device)
                m = m.to(self.device)
                """Scalable Reconstruction"""   

                # if 'normal' in self.args.target:
                #     nout, _, _, _  = self.net_nml(I, M, nImgArray.reshape(-1,1), decoder_resolution=data.data.h * torch.ones(I.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(I.shape[0],1))
                #     nml = (nout * M).squeeze().permute(1,2,0).cpu().detach()
                #     del nout
                
                if 'brdf' in self.args.target:
                    nout, bout, rout, mout  = self.net_brdf(I, M, L, nImgArray.reshape(-1,1), decoder_resolution = data.data.h * torch.ones(I.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(I.shape[0],1))
                    nml = (nout * M).squeeze().permute(1,2,0)
                    base = (bout * M).squeeze().permute(1,2,0)
                    rough = (rout * M).squeeze()
                    metal = (mout * M).squeeze()
                    del nout, bout, rout, mout
                    nml = nml.permute(2,0,1).unsqueeze(0)
                    nml = F.interpolate(nml, size=(r_e-r_s, c_e-c_s), mode='bicubic')
                    nml = nml.squeeze().permute(1,2,0)

                    mask = (torch.abs(1 - torch.sqrt(torch.sum(nml * nml, axis=2))) < 0.5).float()
                    nml = torch.divide(nml, torch.norm(nml, dim=2, keepdim=True) + 1.0e-12)
                    nml = nml * mask[:, :,None]
                    nout = torch.zeros((h_, w_, 3)).to(self.device)
                    nout[r_s:r_e, c_s:c_e,:] = nml

                    base = base.permute(2,0,1).unsqueeze(0)
                    base = F.interpolate(base, size=(r_e-r_s, c_e-c_s), mode='bicubic')
                    base = base.squeeze().permute(1,2,0)
                    bout = torch.zeros((h_, w_, 3)).to(self.device)
                    bout[r_s:r_e, c_s:c_e,:] = base

                    rough = rough.unsqueeze(0).unsqueeze(0)
                    rough = F.interpolate(rough, size=(r_e-r_s, c_e-c_s), mode='bicubic')
                    rough = rough.squeeze().squeeze()
                    rout = torch.zeros((h_, w_)).to(self.device)
                    rout[r_s:r_e, c_s:c_e] = rough

                    metal = metal.unsqueeze(0).unsqueeze(0)
                    metal = F.interpolate(metal, size=(r_e-r_s, c_e-c_s), mode='bicubic')
                    metal = metal.squeeze().squeeze()
                    mout = torch.zeros((h_, w_)).to(self.device)
                    mout[r_s:r_e, c_s:c_e] = metal

                    n_true = N.permute(0,2,3,1).squeeze()
                    mask = (torch.abs(1 - torch.sqrt(torch.sum(n_true * n_true, dim=2))) < 0.5).float()

                    nout = nout * mask[:, :,None]
                    n_true = n_true * mask[:, :,None]
                    bout = bout * mask[:, :,None]
                    a = a * mask[None,None,...]
                    rout = rout * mask
                    r = r * mask[None,None,...]
                    mout = mout * mask
                    m = m * mask[None,None,...]
                    normal_mse = self.mseloss(nout, n_true)/torch.sum(mask)
                    albedo_mse = self.mseloss(bout.permute(2,0,1).unsqueeze(0), a)/torch.sum(mask)
                    rough_mse = self.mseloss(rout.unsqueeze(0).unsqueeze(0), r)/torch.sum(mask)
                    metal_mse = self.mseloss(mout.unsqueeze(0).unsqueeze(0), m)/torch.sum(mask)
                    # nout_cpu = nout.cpu().numpy()
                    # cv2.imwrite(f'{data.data.data_workspace}/normal.png', 255*(0.5 * (1+nout_cpu[:,:,::-1])))
                    loss = (normal_mse + albedo_mse + rough_mse + metal_mse)/4
                    print(f"Normal MSE: {normal_mse:.3f}, Albedo MSE: {albedo_mse:.3f}, Roughness MSE: {rough_mse:.3f}, Metallic MSE: {metal_mse:.3f}, Total MSE: {loss:.3f}")
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.writer.add_scalar('Loss/normal_mse', normal_mse, iter)
                    self.writer.add_scalar('Loss/albedo_mse', albedo_mse, iter)
                    self.writer.add_scalar('Loss/rough_mse', rough_mse, iter)
                    self.writer.add_scalar('Loss/metal_mse', metal_mse, iter)
                    self.writer.add_scalar('Loss/total_mse', loss, iter)



                test_epoch+=1
                if test_epoch % self.test_interval == 0:
                    self.net_brdf.eval()
                    with torch.no_grad():       
                        if self.args.scalable:    
                            patch_size = 512               
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
                                    if 'normal' in self.args.target:
                                        nout, _, _, _  = self.net_nml(pI, pM, nImgArray.reshape(-1,1), decoder_resolution = patch_size * torch.ones(pI.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(pI.shape[0],1))
                                        nout = (F.interpolate(nout, size=(patch_size, patch_size), mode='bilinear', align_corners=True) * pM).cpu()
            
                                    if 'brdf' in self.args.target:
                                        _, bout, rout, mout  = self.net_brdf(pI, pM, nImgArray.reshape(-1,1), decoder_resolution = patch_size * torch.ones(pI.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(pI.shape[0],1))
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
                            if 'normal' in self.args.target:
                                nout, _, _, _  = self.net_nml(I, M, nImgArray.reshape(-1,1), decoder_resolution=data.data.h * torch.ones(I.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(I.shape[0],1))
                                nml = (nout * M).squeeze().permute(1,2,0).cpu().detach()
                                del nout
                            
                            if 'brdf' in self.args.target:
                                _, bout, rout, mout  = self.net_brdf(I, M, nImgArray.reshape(-1,1), decoder_resolution = data.data.h * torch.ones(I.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(I.shape[0],1))
                                base = (bout * M).squeeze().permute(1,2,0).cpu().detach()
                                rough = (rout * M).squeeze().cpu().detach()
                                metal = (mout * M).squeeze().cpu().detach()
                                del bout, rout, mout
                        




                        # save normal of original resolution
                        if 'normal' in self.args.target:
                            nml = nml.cpu().numpy()                
                            nml = cv2.resize(nml, dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_CUBIC)
                            mask = np.float32(np.abs(1 - np.sqrt(np.sum(nml * nml, axis=2))) < 0.5)
                            nml = np.divide(nml, np.linalg.norm(nml, axis=2, keepdims=True) + 1.0e-12)
                            nml = nml * mask[:, :, np.newaxis]
                            nout = np.zeros((h_, w_, 3), np.float32)
                            nout[r_s:r_e, c_s:c_e,:] = nml

                            # if torch.sum(N) > 0:
                            #     n_true = N.permute(0,2,3,1).squeeze().cpu().numpy()
                            #     mask = np.float32(np.abs(1 - np.sqrt(np.sum(n_true * n_true, axis=2))) < 0.5)
                            #     mae, emap = compute_mae.compute_mae_np(nout, n_true, mask = mask)
                            #     print(f"Mean Angular Error (MAE) is {mae:.3f}\n")                        
                            #     emap = emap.squeeze()
                            #     thresh = 90
                            #     emap[emap>=thresh] = thresh
                            #     emap = emap/thresh
                            #     cv2.imwrite(f'{data.data.data_workspace}/error.png', 255*emap)     
                            
                            cv2.imwrite(f'{data.data.data_workspace}/normal.png', 255*(0.5 * (1+nout[:,:,::-1])))                                      

                        if 'brdf' in self.args.target:
                            base = cv2.resize(base.cpu().numpy(), dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_CUBIC)
                            rough = cv2.resize(rough.cpu().numpy(), dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_CUBIC)
                            metal = cv2.resize(metal.cpu().numpy(), dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_CUBIC)

                            bout = np.zeros((h_, w_, 3), np.float32)
                            bout[r_s:r_e, c_s:c_e,:] = base
                            cv2.imwrite(f'{data.data.data_workspace}/baseColor.png', 255*bout[:,:,::-1])

                            rout = np.zeros((h_, w_), np.float32)
                            rout[r_s:r_e, c_s:c_e] = rough
                            cv2.imwrite(f'{data.data.data_workspace}/roughness.png', 255*rout[:,:])

                            mout = np.zeros((h_, w_), np.float32)
                            mout[r_s:r_e, c_s:c_e] = metal
                            cv2.imwrite(f'{data.data.data_workspace}/metallic.png', 255*mout[:,:])
                    self.net_brdf.train()
             



