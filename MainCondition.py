from DiffusionFreeGuidance.TrainCondition import trainLF, evalLF_diffusers, evalLFV2_diffusers


def main(model_config=None):
    modelConfig = {
        "state": "eval",   # or eval
        "epoch": 520,      # training epochs
        "batch_size": 32,   # batch size per card in training & inference batch size
        "T": 1000,         # diffusion steps
        "channel": 64,      
        "channel_mult": [1, 2, 3, 4],
        "num_res_blocks": 4,
        "attn": [2],
        "dropout": 0.15,
        "lr": 1.5e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32*5,
        "grad_clip": 1.,
        "device": "cuda:0",  # not used 
        "w": 3.6,            # classifier free guidance parameter
        "save_dir": "/data/gaors/LFdiffcode/CheckpointsCondition/",    # save pth & load pretrained for test  
        "training_load_weight":  '/data/gaors/LFdiffcode/CheckpointsCondition/ckpt_200.pt',  # keep training load pth
        "test_load_weight": "ckpt_510.pt",
        "sampled_dir": "/data/gaors/LFdiffcode/test_image/results",  # test results
        "sampledImgName": "MMrebuttal.png",
        "nrow": 16,
        "sample_method": 'ddim', # sampling method
        
        # LF dataset args part
        "path_for_train": "/data/gaors/aecode_backup/data_for_training_rgbdisp_16_32/", 
        "data_name": "ALL", # HCI_new, UrbanLF170 or ALL
        "path_for_test":  "/data/gaors/LFdiffcode/test_image/test_h5",
        "data_name_test": "0809x4", 
        "task": "SR"  # not used
    }
    
    if model_config is not None:
        modelConfig = model_config
    
    elif modelConfig["state"] == "train":
        trainLF(modelConfig)

    elif modelConfig["state"] == "eval":
        #evalLF_diffusers(modelConfig)
        evalLFV2_diffusers(modelConfig)  # NYU and DIV2K and INRIA and STFGantry and hciold
    else:
        raise NotImplementedError("Invalid mode!")


if __name__ == '__main__':
    main()
