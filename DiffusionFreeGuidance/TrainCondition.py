

import os
from typing import Dict
import numpy as np
from einops import rearrange
import json
from PIL import Image

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from DiffusionFreeGuidance.DiffusionCondition import GaussianDiffusionTrainerMultiCard
from DiffusionFreeGuidance.LFModelCondition import * 
from DiffusionFreeGuidance.warp import *

from DiffusionFreeGuidance.utils_datasets import TrainSetDataLoaderRGBD, TestSetDataLoaderRGBD, TestSetDataLoaderNYU





def trainLF(modelConfig: Dict):
    from accelerate import Accelerator
    from collections import OrderedDict
    
    accelerator = Accelerator()    

    if not os.path.exists(modelConfig["save_weight_dir"]):
        os.makedirs(modelConfig["save_weight_dir"])
        
    # save training settings
    with open(modelConfig["save_weight_dir"] + 'settings.json', 'w') as f:
        json.dump(modelConfig, f)
    
    # train dataset
    dataset = TrainSetDataLoaderRGBD(modelConfig, warp=True)
    print("Training set length:",len(dataset))
    
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)



    # model setup
    net_model = DistgUNetv3(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])
    

    
    if modelConfig["training_load_weight"] is not None:
        ckpt = torch.load(modelConfig["training_load_weight"], map_location="cuda:0")
        
        ### For multicard trained pth
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            name = k[7:] 
            new_ckpt[name] = v 
        
        net_model.load_state_dict(new_ckpt)
        print("Load ckpt successfully, resume training.")
    
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    
    
    trainer = GaussianDiffusionTrainerMultiCard(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"])

    net_model, optimizer, dataloader, trainer = accelerator.prepare(net_model, optimizer, dataloader, trainer)
    
    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for lr_images, hr_images, depth, warped , _ in tqdmDataLoader:
                
                b = lr_images.shape[0]
                optimizer.zero_grad()
                depth, hr = depth, hr_images  # (b 1 32 32) and (b 3 5*32 5*32)
                
                # Warp condition
                condition = warped               

                if np.random.rand() < 0.1:
                    condition = torch.zeros_like(condition)

                loss = trainer(hr, condition).sum() / b ** 2.
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "hr shape: ": hr.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        cosineScheduler.step()
        if e % 10 == 0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_dir"], 'ckpt_' + str(e) + ".pt"))


def evalLF_diffusers(modelConfig: Dict):
    from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler
    from collections import OrderedDict
    
    device = torch.device(modelConfig["device"])
    
    # set scheduler
    if modelConfig["sample_method"] == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=20)
    elif modelConfig["sample_method"] == "ddpm":
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=700)
    elif modelConfig["sample_method"] == "dpm_solver":
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=100)
    elif modelConfig["sample_method"] == "pndm":
        scheduler = PNDMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=80)
    else:
        raise NotImplementedError
    
    # test dataset
    dataset = TestSetDataLoaderRGBD(modelConfig)
    
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    
    
    # load model and evaluate
    with torch.no_grad():

        # load model
        model = DistgUNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],  # DistgUNetv3
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        
        ### For multicard trained pth
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            name = k[7:] 
            new_ckpt[name] = v 
        
        
        model.load_state_dict(new_ckpt)
        print("model load weight done.")
        model.eval()
        

        count = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for lr_images, hr_images, depth, _ in tqdmDataLoader:
                
                b = lr_images.shape[0]
                #hr_images = hr_images / 255.# if urban_mde, uncomment this line

                depth, hr = depth.to(device), hr_images.to(device)  # (b 1 32 32) and (b 3 5*32 5*32)
                
                center_view =  rearrange(hr, 'b c (u h) (v w) -> b c u v h w', u=5, v=5)[:,:,2,2,:,:] # (b 3 32 32)
                

                
                # Warp condition
                condition = torch.zeros_like(hr)
                for i in range(b):
                    temp = back_projection_from_HR_ref_view(center_view[i,...].unsqueeze(0), depth[i,...].unsqueeze(0).repeat(1,2,1,1)) # (1 uv c h w)
                    condition[i,...] = rearrange(temp, 'b (u v) c h w -> b c (u h) (v w)', u=5, v=5)  # (b 3 5*32 5*32) 
                
                noisyImage = torch.randn_like(hr)                
                
                for i, t in tqdm(enumerate(scheduler.timesteps)):
                    t = t.repeat(b).to(device)
    
                    model_input = scheduler.scale_model_input(noisyImage, t)
         
                    noise_pred = model(model_input, t, condition)
        
                    scheduler_output = scheduler.step(noise_pred, t[0].cpu(), noisyImage)
                    
                    noisyImage = scheduler_output.prev_sample
                

                save_path_ = modelConfig["sampled_dir"] + f'/modelConfig["data_name_test"]'
                if not os.path.exists(save_path_):
                    os.makedirs(save_path_)         
                save_batch_images_to_disk(images_batch=noisyImage, save_path='/data/gaors/DenoisingDiffusionProbabilityModel-ddpm--main/ConditionSampledImgs/exp9_full_dispcorrected/rebuttal-nooverlap-sideboard/', gene_batch=modelConfig["batch_size"], index=count)
                

                count += 1
   

def evalLFV2_diffusers(modelConfig: Dict):
    from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler
    from collections import OrderedDict
    
    device = torch.device(modelConfig["device"])
    
    # set scheduler
    if modelConfig["sample_method"] == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=20)
    elif modelConfig["sample_method"] == "ddpm":
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=1000)
    elif modelConfig["sample_method"] == "dpm_solver":
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=100)
    elif modelConfig["sample_method"] == "pndm":
        scheduler = PNDMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=80)
    else:
        raise NotImplementedError
    
    # test dataset
    dataset = TestSetDataLoaderNYU(modelConfig)
    
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    
    
    # load model and evaluate
    with torch.no_grad():

        # load model
        model = DistgUNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],  
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        
        ### For multicard trained pth
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            name = k[7:] 
            new_ckpt[name] = v 
        
        
        model.load_state_dict(new_ckpt)
        print("model load weight done.")
        model.eval()
        

        count = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for lr_images, hr_images, depth, _ in tqdmDataLoader:
                
                # train
                b, c, h, w = lr_images.shape

                depth, hr = depth.to(device), hr_images.to(device)  # (b 1 32 32) and (b 3 5*32 5*32)
                hr = hr / 255.  
                center_view =  hr
                
                # Warp condition
                condition = torch.zeros((b, 3, 5*h, 5*w))  # b c (u h) (v w)
                for i in range(b):
                    temp = back_projection_from_HR_ref_view(center_view[i,...].unsqueeze(0), depth[i,...].unsqueeze(0).repeat(1,2,1,1)) # (1 uv c h w)
                    condition[i,...] = rearrange(temp, 'b (u v) c h w -> b c (u h) (v w)', u=5, v=5)  # (b 3 5*32 5*32)
                condition = condition.to(device) 
                
                noisyImage = torch.randn(b,3,160,160).to(device) 
                

                for i, t in tqdm(enumerate(scheduler.timesteps)):
                    t = t.repeat(b).to(device)
    
                    model_input = scheduler.scale_model_input(noisyImage, t)
         
                    noise_pred = model(model_input, t, condition)
        
                    scheduler_output = scheduler.step(noise_pred, t[0].cpu(), noisyImage)
                    
                    noisyImage = scheduler_output.prev_sample
                
                save_path_ = modelConfig["sampled_dir"] + f'/{modelConfig["data_name_test"]}'
                if not os.path.exists(save_path_):
                    os.makedirs(save_path_)        
                save_batch_images_to_disk(images_batch=noisyImage, save_path=save_path_, gene_batch=modelConfig["batch_size"], index=count)
                count += 1
  
                                
def save_batch_images_to_disk(images_batch, save_path, gene_batch, index):
    """
    Save a batch of PyTorch tensor images to disk as separate PNG files.
    
    Args:
    - images_batch (torch.Tensor): Batch of images with shape (b, c, h, w).
    - output_dir (str): Directory where the PNG images will be saved.
    """
    b, c, h, w = images_batch.shape
    images_batch = torch.clip(images_batch, 0., 1.)
    output_dir = save_path
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(b):
        image = images_batch[i].cpu().numpy()
        image = image.transpose(1, 2, 0)  # Convert to (h, w, c) format for PIL
        
        # Convert to 8-bit integer (0-255) values
        image = (image * 255).astype('uint8')
        
        image_filename = os.path.join(output_dir, f'image_{index * gene_batch + i}.png')
        image_pil = Image.fromarray(image)
        image_pil.save(image_filename)
        
        print(f'Index {index}, saved image {i+1}/{b} as {image_filename}')