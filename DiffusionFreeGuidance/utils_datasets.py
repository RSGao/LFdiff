import os
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from DiffusionFreeGuidance.utils import *
from einops import rearrange
from DiffusionFreeGuidance.warp import *


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = 5
        self.angRes_out = 5
        if args['task'] == 'SR':
            self.dataset_dir = args['path_for_train'] + 'SR_' + str(5) + 'x' + str(5) + '_' + \
                               str(2) + 'x/'
        elif args['task'] == 'RE':
            self.dataset_dir = args['path_for_train'] + 'RE_' + str(5) + 'x' + str(5) + '_' + \
                               str(5) + 'x' + str(5) + '/'
            pass

        if args['data_name'] == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args['data_name']]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y')) # Lr_SAI_y
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y
            Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y)
            Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        return Lr_SAI_y, Hr_SAI_y, [Lr_angRes_in, Lr_angRes_out]
        #return Lr_SAI_y, Lr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num



class TrainSetDataLoaderRGBD(Dataset):
    def __init__(self, args, warp):
        super(TrainSetDataLoaderRGBD, self).__init__()
        self.warp = warp
        self.angRes_in = 5
        self.angRes_out = 5
        if args['task'] == 'SR':
            self.dataset_dir = args['path_for_train'] + 'SR_' + str(5) + 'x' + str(5) + '_' + \
                               str(2) + 'x/'
        elif args['task'] == 'RE':
            self.dataset_dir = args['path_for_train'] + 'RE_' + str(5) + 'x' + str(5) + '_' + \
                               str(5) + 'x' + str(5) + '/'
            pass

        if args['data_name'] == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args['data_name']]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            #Lr_SAI_y = np.array(hf.get('Lr_SAI_y')) # Lr_SAI_y
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y

            depth = hf.get('disp')
            depth = np.array(depth, dtype='float32') # depth

            #Lr_SAI_y, Hr_SAI_y, depth = augmentation_rgbd(Lr_SAI_y, Hr_SAI_y, depth)
            Hr_SAI_y, depth = augmentation_rgbdv2(Hr_SAI_y, depth)
            
            #Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
            depth = ToTensor()(depth.copy())
            #depth = (depth - depth.min()) / (depth.max() - depth.min()) # normalize depth
            #depth = (depth - (-4)) / 8 # normalize depth

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        

        if self.warp:
            # Warp condition
            center_view = rearrange(Hr_SAI_y, 'c (u h) (v w) -> c u v h w', u=5, v=5)[:,2,2,:,:]  # (3 32 32)
            temp = back_projection_from_HR_ref_view(center_view.unsqueeze(0), depth.unsqueeze(0).repeat(1,2,1,1)) # (1 uv c h w)
            warped = rearrange(temp, 'b (u v) c h w -> b c (u h) (v w)', u=5, v=5).squeeze(0)  # (3 5*32 5*32) 
            return Hr_SAI_y, Hr_SAI_y, depth, warped, [Lr_angRes_in, Lr_angRes_out]
        
        else:
            return Hr_SAI_y, Hr_SAI_y, depth, depth, [Lr_angRes_in, Lr_angRes_out]
        #return Lr_SAI_y, Lr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num


class TestSetDataLoaderRGBD(Dataset):
    def __init__(self, args):
        super(TestSetDataLoaderRGBD, self).__init__()
        self.angRes_in = 5
        self.angRes_out = 5
        if args['task'] == 'SR':
            self.dataset_dir = args['path_for_test'] + 'SR_' + str(5) + 'x' + str(5) + '_' + \
                               str(2) + 'x/'
        elif args['task'] == 'RE':
            self.dataset_dir = args['path_for_train'] + 'RE_' + str(5) + 'x' + str(5) + '_' + \
                               str(5) + 'x' + str(5) + '/'
            pass

        if args['data_name_test'] == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args['data_name_test']]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)
        
        # sort filename
        self.file_list = sorted(self.file_list)  ###
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            #Lr_SAI_y = np.array(hf.get('Lr_SAI_y')) # Lr_SAI_y
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y
            
            depth = np.array(hf.get('disp'), dtype='float32') # depth
            #print(Lr_SAI_y.shape, Hr_SAI_y.shape, depth.shape)
            #Lr_SAI_y, Hr_SAI_y = augmentation_rgbd(Lr_SAI_y, Hr_SAI_y, depth)
            #Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
            depth = ToTensor()(depth.copy())
            #depth = (depth - depth.min()) / (depth.max() - depth.min()) # normalize depth
            #depth = (depth - (-4)) / 8 # normalize depth

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        return Hr_SAI_y, Hr_SAI_y, depth, [Lr_angRes_in, Lr_angRes_out]
        #return Lr_SAI_y, Lr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num



class TestSetDataLoaderNYU(Dataset):
    def __init__(self, args):
        super(TestSetDataLoaderNYU, self).__init__()
        self.angRes_in = 5
        self.angRes_out = 5
        if args['task'] == 'SR':
            self.dataset_dir = args['path_for_test'] + '/'

        if args['data_name_test'] == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args['data_name_test']]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)
        
        # sort filename
        self.file_list = sorted(self.file_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            #Lr_SAI_y = np.array(hf.get('Lr_SAI_y')) # Lr_SAI_y
            rgb = np.array(hf.get('rgb')) # Hr_SAI_y
            
            depth = np.array(hf.get('disp'), dtype='float32') # depth
            #print(Lr_SAI_y.shape, Hr_SAI_y.shape, depth.shape)
            #Lr_SAI_y, Hr_SAI_y = augmentation_rgbd(Lr_SAI_y, Hr_SAI_y, depth)
            #Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            rgb = ToTensor()(rgb.copy())
            depth = ToTensor()(depth.copy())
            #depth = (depth - depth.min()) / (depth.max() - depth.min()) # normalize depth
            #depth = (depth - (-4)) / 8 # normalize depth

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        return rgb, rgb, depth, [Lr_angRes_in, Lr_angRes_out]
        #return Lr_SAI_y, Lr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num



        
def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None
    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        if args.task == 'SR':
            dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.scale_factor) + 'x/'
            data_list = os.listdir(dataset_dir)
        elif args.task == 'RE':
            dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
            self.data_list = [data_name]
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
            Sr_SAI_cbcr = np.array(hf.get('Sr_SAI_cbcr'), dtype='single')
            Lr_SAI_y = np.transpose(Lr_SAI_y, (2, 1, 0)) # (1, 0)
            Hr_SAI_y = np.transpose(Hr_SAI_y, (2, 1, 0)) # (1, 0)
            Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (2, 1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape)==2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1, :]
        label = label[:, ::-1, :]
        #data = data[:, ::-1]        # if gray scale, uncomment bottom two rows 
        #label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :, :]
        label = label[::-1, :, :]
        #data = data[::-1, :]        # if gray scale, uncomment bottom two rows 
        #label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0, 2)
        label = label.transpose(1, 0, 2)
        #data = data.transpose(1, 0,)       # if gray scale, uncomment bottom two rows 
        #label = label.transpose(1, 0)
    return data, label



def augmentation_rgbd(data, label, depth):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1, :]
        label = label[:, ::-1, :]
        depth = np.flip(depth, axis=1)
        #data = data[:, ::-1]        # if gray scale, uncomment bottom two rows 
        #label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :, :]
        label = label[::-1, :, :]
        depth = np.flip(depth, axis=0)
        #data = data[::-1, :]        # if gray scale, uncomment bottom two rows 
        #label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0, 2)
        label = label.transpose(1, 0, 2)
        depth = depth.transpose(1, 0)
        #data = data.transpose(1, 0,)       # if gray scale, uncomment bottom two rows 
        #label = label.transpose(1, 0)
    return data, label, depth
    

def augmentation_rgbdv2(label, depth):
    if random.random() < 0.5:  # flip along W-V direction
        #data = data[:, ::-1, :]
        label = label[:, ::-1, :]
        depth = np.flip(depth, axis=1)
        #data = data[:, ::-1]        # if gray scale, uncomment bottom two rows 
        #label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        #data = data[::-1, :, :]
        label = label[::-1, :, :]
        depth = np.flip(depth, axis=0)
        #data = data[::-1, :]        # if gray scale, uncomment bottom two rows 
        #label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        #data = data.transpose(1, 0, 2)
        label = label.transpose(1, 0, 2)
        depth = depth.transpose(1, 0)
        #data = data.transpose(1, 0,)       # if gray scale, uncomment bottom two rows 
        #label = label.transpose(1, 0)
    return label, depth
       
if __name__ == "__main__":
    import argparse
    modelConfig = {
        "state": "eval", # or eval or generation
        "epoch": 1200,
        "batch_size": 16,
        "T": 1000,
        "channel": 64,
        "channel_mult": [1, 2, 2, 2],
        "attn": [1,2,3],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 2e-4,
        "multiplier": 2.,
        "beta_1": 0.00085,    # 1e-4
        "beta_T": 0.012,    # 0.02
        "img_size": 64*5,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight":  None, #'/data/gaors/DenoisingDiffusionProbabilityModel-ddpm--main/Checkpoints/exp3_diffusersDDPM_DisgUnet/ckpt_599.pt',
        "save_weight_dir": "./Checkpoints/exp5-1_pretrainedVAE_Distg_linear/",
        #"save_weight_dir": "./HCI_new_Checkpoints/new/",
        "test_load_weight": "ckpt_422_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "Noisy_epoch599_1st.png",
        #"sampledImgName": "HCInew_exp2_test1.png", 
        #"sampledImgName": "exp3_diffusers_DisgUnet_sample1_948e__ddpm.png",
        "sampledImgName": "exp5-1_pretrainedVAE_Distg_linear_sample1_422e_ddpm.png",
        "nrow": 8,
        "sample_method": "ddpm",   # now support: ddpm, ddim, dpm_solver, pndm
        
        # LF dataset args part
        "path_for_train": "/data/gaors/aecode_backup/data_for_training_rgbdisp_64_128/", 
        "path_for_test":  "/data/gaors/aecode_backup/data_for_test_rgbd_16_32/",
        "data_name": "UrbanLF", # UrbanLF
        "data_name_test": "HCI_new_4", # HCI_new_4
        "task": "SR"
        
        }


    train = TrainSetDataLoaderRGBD(modelConfig, warp=False)
    print(len(train))
    Hr_SAI_y, Hr_SAI_y, depth, _, [Lr_angRes_in, Lr_angRes_out] = train[186]
    print(Hr_SAI_y)
    save_image(Hr_SAI_y.unsqueeze(0), 'hr.png', nrow=modelConfig["nrow"])
    #save_image(Lr_SAI_y.unsqueeze(0), 'lr.png', nrow=modelConfig["nrow"])
    save_image(depth.unsqueeze(0), 'depth.png', nrow=modelConfig["nrow"])
    #print(Lr_SAI_y.shape)
    print(Hr_SAI_y.shape)
    print(depth.shape)
