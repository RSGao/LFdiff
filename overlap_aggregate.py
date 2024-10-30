import cv2
import numpy as np
import os
from einops import rearrange

def get_uv(u, v, input_path, index):
    H, W = 512, 512  # modify this
    root_path = input_path
    path = sorted(os.listdir(root_path), key=lambda x:int(x[6:-4]))


    crop_size = (32, 32)  # modify this
    stride = 16           # modify this

    crop_count_x = (W - crop_size[1]) // stride + 1
    crop_count_y = (H - crop_size[0]) // stride + 1

    result_size = (H, W)
    print(result_size)

    result_image = np.zeros((result_size[1], result_size[0], 3), dtype=np.float64)
    count_image = np.zeros((result_size[1], result_size[0]), dtype=np.uint8)

    count = 0

    for x in range(0, H - crop_size[1] + 1, stride):
        for y in range(0, W - crop_size[0] + 1, stride):
            print(root_path + '/' + path[count])
            small_image = cv2.imread(root_path + '/' + path[count]) # (uv hw 3)
            small_image =  rearrange(small_image, '(u h) (v w) c -> v w u h c', u=5, v=5)[v,:,u,:,:]  # u v

            result_x = x
            result_y = y

            result_image[result_y:result_y + crop_size[0], result_x:result_x + crop_size[1]] += small_image
            count_image[result_y:result_y + crop_size[0], result_x:result_x + crop_size[1]] += 1
            count += 1

    count_image[count_image == 0] = 1

    result_image = (result_image / count_image[:, :, np.newaxis]).astype(np.uint8)
    result_image = np.rot90(result_image, k=3, axes=(0, 1))
    result_image = np.flip(result_image, axis=1)

    savepath = f'/data/gaors/LFdiffcode/test_image/results/{index}_fullres'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    cv2.imwrite(savepath + f'/view_{u}_{v}.png', result_image)


if __name__ == "__main__":
    indexes = ['0809x4']
    
    for index in indexes:
        input_path = f'/data/gaors/LFdiffcode/test_image/results/{index}'
        print(input_path)
        for u in range(5):
            for v in range(5):
                get_uv(u, v, input_path, index)
    
