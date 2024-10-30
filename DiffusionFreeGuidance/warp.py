import torch
import torch.nn as nn
import numpy as np

def back_projection_from_HR_ref_view(sr_ref, disparity, refPos= [2,2], angular_resolution=5, scale=1, padding_mode="reflection"):
    # sr_ref: [B, 1, H, W]
    # refPos: [u, v]
    # disparity: [B, 2, h, w]
    # angular_resolution: U
    UV = angular_resolution * angular_resolution
    B = sr_ref.shape[0]
    C = sr_ref.shape[1]
    ref_u = refPos[1] # horizontal angular coordinate
    ref_v = refPos[0] # vertical angular coordinate
    ## generate angular grid
    # x here denotes the horizontal line
    # so uu here also denotes the horizontal line (column number)
    uu = torch.arange(0, angular_resolution).view(1, -1).repeat(angular_resolution, 1) # u direction, X
    vv = torch.arange(0, angular_resolution).view(-1, 1).repeat(1, angular_resolution) # v direction, Y
    uu = uu.view(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1) - ref_u
    vv = vv.view(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1) - ref_v
    # uu = arange_uu - ref_u
    # vv = arange_vv - ref_v
    deta_uv = torch.cat([uu, vv], dim=2) # [B, U*V, 2, 1, 1]
    if sr_ref.is_cuda:
        deta_uv = deta_uv.cuda()
    deta_uv = deta_uv.float()
    ## generate the full disparity maps
    full_disp = disparity.unsqueeze(1) # [B, 1, 2, h, w]
    full_disp = full_disp.repeat(1, UV, 1, 1, 1) # [B, U*V, 2, h, w]
    full_disp = full_disp * deta_uv # [B, U*V, 2, h, w]

    # repeat sr_ref
    sr_ref = sr_ref.repeat(1, UV, 1, 1, 1) # [B, U*V, 1, H, W]

    # view
    full_disp = full_disp.view(-1, 2, full_disp.shape[3], full_disp.shape[4])
    sr_ref = sr_ref.view(-1, C, sr_ref.shape[3], sr_ref.shape[4])

    # output the back-projected light fields
    bp_lr_lf = warp_back_projection_no_range(sr_ref, full_disp, scale, padding_mode=padding_mode) # [BUV, C, h, w]
    bp_lr_lf = bp_lr_lf.view(-1, UV, C, bp_lr_lf.shape[2], bp_lr_lf.shape[3])
    return bp_lr_lf



def warp_back_projection_no_range(x, flo, scale, padding_mode="zeros"):
    """
    sample the points from HR images with LR flow for back-projection.

    x: [B, C, H, W] HR image
    flo: [B, 2, h, w] LR_flow
    x and flo should be inside the same device (CPU or GPU)

    """
    # B, C, H, W = x.shape
    B, _, H, W = flo.shape
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = grid - flo

    # make coordinate transformation from LR to HR
    vgrid = coordinate_transform(vgrid, 1.0/scale)

    # scale grid to [-1,1]
    vgridx = vgrid[:, 0, :, :].clone().unsqueeze(1)
    vgridy = vgrid[:, 1, :, :].clone().unsqueeze(1)
    vgridx = 2.0 * vgridx / max(W * scale - 1, 1) - 1.0
    vgridy = 2.0 * vgridy / max(H * scale - 1, 1) - 1.0

    vgrid = torch.cat([vgridx, vgridy], dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode=padding_mode)

    return output

def coordinate_transform(x, scale):
    # x can be tensors with any dimensions
    # scale is the scaling factors, when it's less than 1, HR2LR, when it's larger than 1, LR2HR
    y = x / scale - 0.5 * (1 - 1.0 / scale) # for python coordinate system
    return y

if __name__ == "__main__":
    import torchvision.transforms as tf
    import cv2
    from einops import rearrange
    from torchvision.utils import save_image
    #x = torch.randn(1, 3, 32, 32)
    #depth = torch.randn(1, 1, 32, 32).repeat(1, 2, 1, 1)
    #print(depth.shape)
    x = cv2.imread('/data/gaors/DenoisingDiffusionProbabilityModel-ddpm--main/ConditionSampledImgs/center_view.png')
    depth = cv2.imread('/data/gaors/DenoisingDiffusionProbabilityModel-ddpm--main/ConditionSampledImgs/depth_old.png', 0)
    
    x2 = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    tot = tf.ToTensor()
    x, x2, depth = tot(x).unsqueeze(0), tot(x2).unsqueeze(0), tot(depth).unsqueeze(0)
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    batch = torch.zeros(2, 3, 32, 32)
    batchd = torch.zeros(2, 1, 32, 32)
    
    batch[0] = x.clone()
    batch[1] = x2.clone()
    
    batchd[0] = depth.clone()
    batchd[1] = depth.clone()
    
    print(batch.shape, batchd.shape)
    

    warped = back_projection_from_HR_ref_view(batch, batchd)
    print(warped.shape)
    warped = rearrange(warped[0].unsqueeze(0), 'b (u v) c h w -> b c (u h) (v w)', u=5, v=5)
    #print(warped.shape)
    #save_image(x, 'x.png')
    #save_image(x2, 'x2.png')
    save_image(warped, 'warped.png')