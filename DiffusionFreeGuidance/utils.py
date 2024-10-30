import numpy as np
import os
#from skimage import metrics
import torch
from pathlib import Path
#import matplotlib.pyplot as plt
import logging
from einops import rearrange
import torch.nn.functional as F


def make_pairs(l, t1, t2, num_pairs, given_vid):
    B, T, C, H, W = given_vid.size()
    idx1 = t1.view(B, num_pairs, 1, 1, 1, 1).expand(B, num_pairs, 1, C, H, W).type(torch.int64)
    frame1 = torch.gather(given_vid.unsqueeze(1).repeat(1,num_pairs, 1,1,1,1), 2, idx1).squeeze()
    t1 = t1.float() / (l - 1) 

    idx2 = t2.view(B, num_pairs, 1, 1, 1, 1).expand(B, num_pairs, 1, C, H, W).type(torch.int64)
    frame2 = torch.gather(given_vid.unsqueeze(1).repeat(1,num_pairs,1,1,1,1), 2, idx2).squeeze()
    t2 = t2.float() / (l - 1) 

    frame1 = frame1.view(-1, C, H, W)
    frame2 = frame2.view(-1, C, H, W)

    # sort by t
    t1 = t1.view(-1, 1, 1, 1).repeat(1, C, H, W)
    t2 = t2.view(-1, 1, 1, 1).repeat(1, C, H, W)

    ret_frame1 = torch.where(t1 < t2, frame1, frame2)
    ret_frame2 = torch.where(t1 < t2, frame2 ,frame1)

    t1 = t1[:, 0:1]
    t2 = t2[:, 0:1]

    ret_t1 = torch.where(t1 < t2, t1, t2)
    ret_t2 = torch.where(t1 < t2, t2, t1)

    dt = ret_t2 - ret_t1

    return torch.cat([ret_frame1, ret_frame2, dt], dim=1)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count






def cal_metrics(args, label, out,):
    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    for b in range(B):
        for u in range(U):
            for v in range(V):
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_y[b, u, :, v, :].numpy(), out_y[b, u, :, v, :].numpy(), data_range=1.)
                if args.task == 'RE':
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True,
                                                                  sigma=1.5, use_sample_covariance=False, data_range=1.)
                else:
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True, data_range=1.)
                pass

    if args.task=='RE':
        for u in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
            for v in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
                PSNR[:, u, v] = 0
                SSIM[:, u, v] = 0

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out

def LFdivide_rgb(data, angRes, patch_size, stride):
    data = rearrange(data, 'c (a1 h) (a2 w) -> (a1 a2) c h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()
    print(data.shape)
    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    # pad = torch.nn.ReflectionPad2d(padding=(bdr, bdr+stride-1, bdr, bdr+stride-1))
    # data_pad = pad(data)
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    print(subLF.shape)
    subLF = rearrange(subLF, '(a1 a2) c (h w) (n1 n2) -> n1 n2 c (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subLF
    
def LFintegrate_rgb(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 5:
        subLF = rearrange(subLF, 'n1 n2 c (a1 h) (a2 w) -> n1 n2 c a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 c a1 a2 h w -> a1 a2 c (n1 h) (n2 w)')
    outLF = outLF[:, :, :, 0:h, 0:w]

    return outLF
    
def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    # pad = torch.nn.ReflectionPad2d(padding=(bdr, bdr+stride-1, bdr, bdr+stride-1))
    # data_pad = pad(data)
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)
    return subLF


def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y
