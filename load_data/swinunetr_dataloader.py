
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandFlipd,
    RandCropByPosNegLabeld,
    NormalizeIntensityd,    
    RandRotated,
    SqueezeDimd,
    SpatialPadd,
    RandSpatialCropd,
    RandAdjustContrastd,
    RandAffined,
    RandRotated,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandScaleIntensityFixedMeand,
    RandSimulateLowResolutiond,
    RandZoomd,
)

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist
)

import numpy as np

from torch.utils.data import ConcatDataset
from os.path import join

class dataloader(object):
    def __init__(self, dims, json, fold, model, num_samples = 4, batch_size=1, cachenum = 1, val_cachenum=1, patchsize=96):
        self.dims = dims
        self.json = json
        self.fold = fold
        self.model = model
        self.batch_size = batch_size
        self.cachenum = cachenum
        self.valcachenum = val_cachenum
        self.patchsize = patchsize
        if self.model == 'mod5':
            keys = ["image", "label", "normcoord"] #"sinenc",
            interp_mode = ['bilinear', 'nearest', 'bilinear'] 
        else:
            keys = ["image", "label"]
            interp_mode = interp_mode = ['bilinear', 'nearest']

        spacing = (50,50)
        mag2d = [1., 5.]
        mag3d = [50., 500.]
        sigma = [9., 15.]
        scale_range = [0.7, 1.4]
        rotation_x = [np.deg2rad(-90), np.deg2rad(90)]
        rotation_y = [np.deg2rad(-90), np.deg2rad(90)]
        rotation_z = [np.deg2rad(-90), np.deg2rad(90)]
        rotation_x_2d = [np.deg2rad(-90), np.deg2rad(90)]
        gamma_range = (0.7, 1.5)
        sigma_smooth = (0.5,1)
        intensity_scale = (-0.25, 0.25)
        zoom_range = (0.5, 1.0)

        prob_DA = 0.5

        prob_xfms = {
            "prob_simulowres": 0.25,
            "prob_smooth": 0.2,
            "prob_noise": 0.1,
            "prob_gamma": 0.3,
            "prob_gamma_inv": 0.1,
            "prob_inten_scale": 0.15,
            "prob_inten_fixedmu": 0.15,
            "prob_flip_0": 1,
            "prob_flip_1": 0.5,
            "prob_flip_2": 0.5,
            "prob_rotate": 0.2,
            "prob_trans": 0.4,
            "prob_zoom": 0.2
        }

        prob_DA_xfms = {}
        for p in prob_xfms.keys():
            prob_DA_xfms[p] = prob_DA * prob_xfms[p]

        if self.dims == 2:
            self.roisize = (self.patchsize,self.patchsize)
            self.train_transforms = Compose(
                [
                    LoadImaged(keys=keys),
                    EnsureChannelFirstd(keys=keys),
                    RandSimulateLowResolutiond(keys='image',
                                            prob = prob_DA_xfms['prob_simulowres'],
                                            zoom_range=zoom_range),
                    RandGaussianSmoothd(keys='image', 
                                        prob=prob_DA_xfms['prob_smooth'],
                                        sigma_x=sigma_smooth,
                                        sigma_z=sigma_smooth),
                    RandGaussianNoised(keys='image', 
                                        prob=prob_DA_xfms['prob_noise'],
                                        ),
                    RandAdjustContrastd(keys='image',
                                        gamma = gamma_range,
                                        retain_stats = True,
                                        prob = prob_DA_xfms['prob_gamma']
                                    ),
                    RandAdjustContrastd(keys='image',
                                        gamma = gamma_range,
                                        retain_stats = True,
                                        prob = prob_DA_xfms['prob_gamma_inv'],
                                        invert_image=True
                                    ),
                    RandScaleIntensityd(keys="image", 
                                        prob=prob_DA_xfms['prob_inten_scale'],
                                        factors=intensity_scale),
                    RandScaleIntensityFixedMeand(keys="image", 
                                        prob=prob_DA_xfms['prob_inten_fixedmu'],
                                        factors=intensity_scale,
                                        preserve_range = True
                    ),
                    NormalizeIntensityd(keys="image", 
                                        nonzero=True),
                    RandFlipd(keys = keys,
                        spatial_axis = 0, prob = prob_DA_xfms['prob_flip_0'],
                    ),
                    RandFlipd(keys = keys,
                        spatial_axis = 1, prob = prob_DA_xfms['prob_flip_1'],
                    ),
                    RandFlipd(keys = keys,
                        spatial_axis = 2, prob = prob_DA_xfms['prob_flip_2'],
                    ),
                    RandSpatialCropd(keys=keys,roi_size=[-1,1,-1],
                                            random_size = False),
                    SqueezeDimd(keys=keys, dim=2),
                    RandRotated(keys=keys,
                                prob = prob_DA_xfms['prob_rotate'],
                                range_x = rotation_x_2d,
                                mode = interp_mode,
                                padding_mode = 'zeros',
                    ),
                    RandAffined(keys=keys,
                                prob=prob_DA_xfms['prob_trans'],
                                translate_range = ((-50,50),(-50,50)),
                                mode = interp_mode,
                                padding_mode = 'zeros',
                                ),
                    RandZoomd(keys=keys,
                            prob=prob_DA_xfms['prob_zoom'],
                            min_zoom=scale_range[0],
                            max_zoom=scale_range[1],
                            mode = interp_mode,
                            padding_mode='constant'
                            ),
                    SpatialPadd(keys=keys,
                                spatial_size=self.roisize),
                    RandCropByPosNegLabeld(
                        keys=keys,
                        label_key= "label",
                        spatial_size=self.roisize,
                        pos=1,
                        neg=1,
                        num_samples=3
                    ),
                ]
            )
        else:
            self.roisize = (self.patchsize,self.patchsize, self.patchsize)
            self.train_transforms = Compose(
                [
                    LoadImaged(keys=keys),
                    EnsureChannelFirstd(keys=keys),
                    RandSimulateLowResolutiond(keys='image',
                                            prob = prob_DA_xfms['prob_simulowres'],
                                            zoom_range=zoom_range),
                    RandGaussianSmoothd(keys='image', 
                                        prob=prob_DA_xfms['prob_smooth'],
                                        sigma_x=sigma_smooth,
                                        sigma_y=sigma_smooth,
                                        sigma_z=sigma_smooth),
                    RandGaussianNoised(keys='image', 
                                        prob=prob_DA_xfms['prob_noise'],
                                        ),
                    RandAdjustContrastd(keys='image',
                                        gamma = gamma_range,
                                        retain_stats = True,
                                        prob = prob_DA_xfms['prob_gamma']
                                    ),
                    RandAdjustContrastd(keys='image',
                                        gamma = gamma_range,
                                        retain_stats = True,
                                        prob = prob_DA_xfms['prob_gamma_inv'],
                                        invert_image=True
                                    ),
                    RandScaleIntensityd(keys="image", 
                                        prob=prob_DA_xfms['prob_inten_scale'],
                                        factors=intensity_scale),
                    RandScaleIntensityFixedMeand(keys="image", 
                                        prob=prob_DA_xfms['prob_inten_fixedmu'],
                                        factors=intensity_scale,
                                        preserve_range = True
                    ),
                    NormalizeIntensityd(keys="image", 
                                        nonzero=True),
                    RandFlipd(keys = keys,
                        spatial_axis = 0, prob = prob_DA_xfms['prob_flip_0'],
                    ),
                    RandFlipd(keys = keys,
                        spatial_axis = 1, prob = prob_DA_xfms['prob_flip_1'],
                    ),
                    RandFlipd(keys = keys,
                        spatial_axis = 2, prob = prob_DA_xfms['prob_flip_2'],
                    ),
                    RandRotated(keys=keys,
                                prob = prob_DA_xfms['prob_rotate'],
                                range_x = rotation_x,
                                range_y = rotation_y,
                                range_z = rotation_z,
                                mode = interp_mode,
                                padding_mode = 'zeros',
                    ),
                    RandAffined(keys=keys,
                                prob=prob_DA_xfms['prob_trans'],
                                translate_range = ((-50,50),(-50,50),(-50,50)),
                                mode = interp_mode,
                                padding_mode = 'zeros',
                                ),
                    RandZoomd(keys=keys,
                            prob=prob_DA_xfms['prob_zoom'],
                            min_zoom=scale_range[0],
                            max_zoom=scale_range[1],
                            mode = interp_mode,
                            padding_mode='constant'
                            ),
                    SpatialPadd(keys=keys,
                                spatial_size=self.roisize),
                    RandCropByPosNegLabeld(
                        keys=keys,
                        label_key= "label",
                        spatial_size=self.roisize,
                        pos=1,
                        neg=1,
                        num_samples=1
                    ),
                ]
            )

        self.val_transforms = Compose(
            [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                NormalizeIntensityd(keys="image", 
                                    nonzero=True),
            ]
        )

    def prep_data(self, cachedataWorkers=2, dataLoaderWorkers=2):
        train_files = load_decathlon_datalist(self.json, True, "train_"+str(self.fold))
        val_files = load_decathlon_datalist(self.json, True, "val_"+str(self.fold))
        
        cachedataWorkers = cachedataWorkers
        dataLoaderWorkers = dataLoaderWorkers
        train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_num=self.cachenum,
            cache_rate=1.0,
            num_workers=cachedataWorkers,
        )
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=dataLoaderWorkers, pin_memory=True)

        val_ds = CacheDataset(data=val_files, transform=self.val_transforms, cache_num=self.valcachenum, cache_rate=1.0, num_workers=cachedataWorkers)
        
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=dataLoaderWorkers, pin_memory=True)
        
        return train_loader, val_loader
