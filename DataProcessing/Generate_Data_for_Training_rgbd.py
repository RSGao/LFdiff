import argparse
import os
import h5py
from imresize import *
from pathlib import Path
import scipy.io as scio
import sys
import cv2
from read_pfm import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--scale_factor", type=int, default=2, help="4, 2")
    parser.add_argument('--data_for', type=str, default='test_single_rgb', help='')
    parser.add_argument('--src_data_path', type=str, default='/data/gaors/aecode_backup/dataset2/', help='')
    parser.add_argument('--save_data_path', type=str, default='/data/gaors/aecode_backup/data_for_test_single_rgb/SR_5x5_2x/', help='')

    return parser.parse_args()


def main(args):
    angRes, scale_factor = args.angRes, args.scale_factor
    patchsize = 32
    stride = 16
    downRatio = 1 / scale_factor

    ''' dir '''
    save_dir = Path(args.save_data_path)
    save_dir.mkdir(exist_ok=True)


    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['HCI_new']:
            continue
        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)

        src_sub_dataset = args.src_data_path + name_dataset + '/training/' 
        
        for root, dirs, files in os.walk(src_sub_dataset):
            print('here')
            for file in files:
                idx_scene_save = 0
                print('Generating training data of Scene_%s in Dataset %s......\t' %(file, name_dataset))
                try:
                    data = h5py.File(root + file, 'r')
                    LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                    depth = read_pfm('/data/gaors/aecode_backup/disps/HCI_new' + file[:-4] + '.pfm')
                except:
                    data = scio.loadmat(root + file)['LF']
                    LF = np.array(data).transpose((4, 3, 2, 1, 0)) # for padded (EPFL and Inria)
                    depth = read_pfm('/data/gaors/aecode_backup/disps/HCI_new' + file[:-4] + '.pfm')

                (U, V, _, _, _) = LF.shape    # (9 9 512 512 3)
                print(LF.shape)

                # Extract central angRes * angRes views
                LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, :, :, 0:3]
                LF = LF.astype('double')
                (U, V, H, W, _) = LF.shape    # (5 5 512 512 3)

                for h in range(0, H - patchsize + 1, stride):
                    for w in range(0, W - patchsize + 1, stride):
                        idx_save = idx_save + 1
                        idx_scene_save = idx_scene_save + 1
                        Hr_SAI_y = np.zeros((U * patchsize, V * patchsize, 3),dtype='single')
                        Lr_SAI_y = np.zeros((U * patchsize // scale_factor, V * patchsize // scale_factor, 3),dtype='single')

                        for u in range(U):
                            for v in range(V):
                                tmp_Hr_rgb = LF[u, v, h: h + patchsize, w: w + patchsize,:]
                                tmp_Hr_y = tmp_Hr_rgb

                                patchsize_Lr = patchsize // scale_factor
                                Hr_SAI_y[u * patchsize : (u+1) * patchsize, v * patchsize: (v+1) * patchsize, :] = tmp_Hr_y
                                tmp_Sr_y = imresize(tmp_Hr_y, scalar_scale=downRatio)

                                Lr_SAI_y[u*patchsize_Lr : (u+1)*patchsize_Lr, v*patchsize_Lr: (v+1)*patchsize_Lr, :] = tmp_Sr_y
                                pass
                            pass

                        # depth
                        depth_patch = depth[h: h + patchsize, w: w + patchsize]        
                        # save
                        file_name = [str(sub_save_dir) + '/' + '%06d'%idx_save + '.h5']
                        with h5py.File(file_name[0], 'w') as hf:
                            hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y.transpose((0, 1, 2)), dtype='single')
                            hf.create_dataset('Hr_SAI_y', data=Hr_SAI_y.transpose((0, 1, 2)), dtype='single')
                            hf.create_dataset('disp', data=depth_patch.transpose((0, 1)), dtype='float')
                            hf.close()
                            pass

                        pass
                    pass
                #
                print('%d training samples have been generated\n' % (idx_scene_save))

                pass
            pass
        pass

    pass



if __name__ == '__main__':
    args = parse_args()

    main(args)