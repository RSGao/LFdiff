import os
import h5py
from PIL import Image
import numpy as np


input_dir = '/data/gaors/LFdiffcode/test_image'
rgb_dir = os.path.join(input_dir, 'rgb')
depth_dir = os.path.join(input_dir, 'rescaled_disp')


rgb_files = os.listdir(rgb_dir)
depth_files = os.listdir(depth_dir)


if len(rgb_files) != len(depth_files):
    print("Number of RGBs != number of depths")

else:
    patch_size = (32, 32)  
    stride = 16  
    num_patches = ((patch_size[0] - 1) // stride + 1) * ((patch_size[1] - 1) // stride + 1)
    idx_save = 0

    for i in range(len(rgb_files)):

        rgb_path = os.path.join(rgb_dir, rgb_files[i])
        depth_path = os.path.join(depth_dir, depth_files[i])
        rgb_name = rgb_files[i].split('.')[0]
        h5_path = '/data/gaors/LFdiffcode/test_image/test_h5' + f'/{rgb_name}/'
        if not os.path.exists(h5_path):  
            os.makedirs(h5_path)         

        print(f'processing image {i+1} / {len(rgb_files)}')
        rgb_image = Image.open(rgb_path)
        depth_image = np.load(depth_path)
        depth_image = Image.fromarray(depth_image)
        
        
        for y in range(0, rgb_image.height - patch_size[0] + 1, stride):
            for x in range(0, rgb_image.width - patch_size[1] + 1, stride):

                rgb_patch = rgb_image.crop((x, y, x + patch_size[1], y + patch_size[0]))
                depth_patch = depth_image.crop((x, y, x + patch_size[1], y + patch_size[0]))

                rgb_data = np.array(rgb_patch)[:,:,0:3]
                depth_data = np.array(depth_patch)
                
                file_name = h5_path + '%06d'%idx_save + '.h5'

                with h5py.File(file_name, 'w') as hf:
                    hf.create_dataset('rgb', data=rgb_data.transpose((0, 1, 2)), dtype='single')
                    hf.create_dataset('disp', data=depth_data.transpose((0, 1)), dtype='float')
                    hf.close()
                    pass
                idx_save += 1

    print("Done.")
