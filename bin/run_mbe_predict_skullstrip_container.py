#!/usr/bin/env python3

from __future__ import unicode_literals, print_function
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# comment out if you would like warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib

from monai.inferers import sliding_window_inference,SliceInferer

from run_mbe_skullstrip_pp import post_process, smooth_label
from prep.generate_posencs import sphenc, norm_coords, norm_sinenc

from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    NormalizeIntensityd,
    RandRotate90d,
    RandRotated,
    SqueezeDimd,
    SplitDimd,
    SpatialPadd,
    SpatialPad
)

from monai.data import (
    decollate_batch,
    load_decathlon_datalist,
    CacheDataset
)

import argparse
# from diceCoeff import diceCoeff
# from medpy import metric

import torch
from shutil import copyfile


def parser():
    parser = argparse.ArgumentParser(description='Skull stripping using Mouse Brain Extractor (container version). This script is to be used for containers. \nCompulsory arguments: (-i) and (-o) MUST be provided.')
    parser.add_argument('--j', help="Input JSON file name for batch processing.", required=False)
    parser.add_argument('-i', help="Input MRI file name.", required=True)
    parser.add_argument('-o', help="Output file name for mask.", required=True)
    parser.add_argument('-d', help='Number of dimensions.', required=False, type=int, choices=[2,3], default=3)
    parser.add_argument('-m', help='Model type.', required = False, choices=['mod5', 'orig'], default='mod5')
    parser.add_argument('-n', help='Model file that contains pre-trained weights.', required=False)
    parser.add_argument('-p', help="If global absolute positional encodings (GPEs) have already been generated, you can specify the file using this flag.", required=False)
    parser.add_argument('-b', help="Number of ROIs processed at once.", required=False, default=4, type=int)
    parser.add_argument('--key', help="If json file is provided, the dictionary key to process.", required=False)
    parser.add_argument('--no_pp', help="Do not run post-processing on the masks.", required=False, action='store_true')
    parser.add_argument('--copy_input', help="Copy input to output directory for easy access.", required=False, action='store_true')
    parser.add_argument('--dstype', help="In vivo or ex vivo. Arugment for RAP BIDS App.", choices=['invivo_iso','invivo_aniso', 'exvivo'], required=False, default='invivo_iso', type=str)
    parser.add_argument('--strip', help='Segment the input file with the output mask. If JSON file is specified for -i, then place specify "True" as argument value (e.g., --strip True).', required=False)
    parser.add_argument('--gen_posenc', help="Generate positional encoding (GPE) files prior to running prediction.", required=False, action='store_true')
    parser.add_argument('--posenc', help="What you would like to name your GPE file.", required=False)
    parser.add_argument('--device', help="Specify device for processing (cpu or cuda/gpu). If gpu is specified but is not available, then cpu will be used.", 
                        choices=['cpu','cuda'], default='cuda',required=False)
    parser.add_argument('--patch_size', help='Patch size. One int number should be given. Patch size is isotropic.', default=96, type=int)
    return parser

def main():
    args = parser().parse_args()

    if (not args.i) and (not args.j):
        print('Either and input image file name or json file containing the input file names must be provided.')
        sys.exit(2)
    if (args.j) and (not args.key):
        print('If JSON file is provided, then the dictionary key that has the values of the file names must be provided.')
        sys.exit(2)

    use_pos = False
    if args.m == 'mod5':
        from models.swin_unetr_mod5 import SwinUNETR
        use_pos = True
    elif args.m == 'orig':
        from models.swin_unetr import SwinUNETR
    else:
        print('Model is invalid. Please choose an available model.')
        sys.exit(2)
    
    if args.n:
        modelname = args.n
    else:
        modelname = f'/mod5/{args.dstype}/checkpoint_best.pth'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.device == 'cpu') or (not torch.cuda.is_available()):
        device = torch.device("cpu")
        args.device = "cpu"
    
    if args.dstype == 'invivo_aniso':
        spatial_dims = 2
        patch_size = 128
        imgsize = (patch_size, patch_size)
        coronal_inferer = SliceInferer(
        roi_size=imgsize,
        sw_batch_size=args.b,
        spatial_dim=1, 
        cval=0,
        progress=False,
        sw_device=args.device, 
        device="cpu",
        overlap=0.8,
    )
    else:
        spatial_dims = 3
        patch_size = 96
        imgsize = (patch_size, patch_size, patch_size)

    num_classes = 2

    model = SwinUNETR(
        img_size=imgsize,
        in_channels=1,
        out_channels=num_classes,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=spatial_dims
    ).to(device)

    # modelname = args.n
    weight = torch.load(modelname, map_location=device)
    model.load_from(weights=weight)
    model.load_state_dict(torch.load(modelname, map_location=device))

    cachedataWorkers=2

    keys = ["image"]
    
    if args.m == 'mod5':
        # pos = "sinenc"
        pos = "normcoord"
        keys = keys + [pos]

    val_transforms = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            NormalizeIntensityd(keys="image", 
                                nonzero=True),
        ]
    )

    ## get input file names
    if args.i:
        T2_input = args.i
        file_list = [{
        "image": T2_input,
        },]
        if use_pos:
            if args.p:
                posfile = args.p
            else:
                basename = args.o.split('.nii')[0]
                posfile = basename + '.gpe.nii.gz'
                if args.m == 'mod5':
                    norm_coords(T2_input, posfile)
            file_list[0].update({
                pos:posfile
            })
        outputs = [args.o]
        output_folder = os.path.dirname(args.o)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if args.strip:
            stripped = [args.strip]

    elif args.j:
        file_list = load_decathlon_datalist(args.j, True, args.key)
        outputs = [f"{args.o}/{file_list[i]['image'].split('/')[-1].split('.nii')[0]}.output.nii.gz" for i in range(len(file_list))]
        if args.strip:
            stripped = [f"{args.o}/{file_list[i]['image'].split('/')[-1].split('.nii')[0]}.stripped.nii.gz" for i in range(len(file_list))]
        output_folder = args.o
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    ds = CacheDataset(data=file_list, transform=val_transforms, cache_num=1, cache_rate=1.0, num_workers=cachedataWorkers)

    torch.backends.cudnn.benchmark = True
    model.eval()
    
    batch = args.b
    for idx in range(len(ds)):
        with torch.no_grad():
            img = ds[idx]["image"]
            print("\nProcessing", img.meta['filename_or_obj'])
            if args.copy_input:
                if not os.path.exists(output_folder+'/T2w/'):
                    os.mkdir(output_folder+'/T2w/')
                copyfile(img.meta['filename_or_obj'], output_folder+'/T2w/'+img.meta['filename_or_obj'].split('/')[-1])
            val_inputs = torch.unsqueeze(img, 1)
            if use_pos:
                if args.m == 'mod5':
                    normcoords = ds[idx][pos].unsqueeze(0)
                    global_pos = norm_sinenc(normcoords, normcoords.shape)
                inputs = torch.cat([val_inputs,global_pos], 1)
            else:
                inputs = val_inputs
            
            if spatial_dims == 3:
                val_outputs = sliding_window_inference(inputs, imgsize, batch, model, overlap=0.8, sw_device=args.device, device="cpu", progress=True)
            else:
                val_outputs = coronal_inferer(inputs=inputs, network=model) 
            
            pred = torch.argmax(val_outputs, dim=1).detach().cpu().squeeze(0).squeeze(0).numpy()
            # gt = val_labels.cpu().squeeze(0).squeeze(0).numpy()

        
        # orig_shape = np.zeros(ds[idx]["image"].meta["spatial_shape"])
        # fg_start = ds[idx]["foreground_start_coord"]
        # fg_end = ds[idx]["foreground_end_coord"]
        # orig_shape[fg_start[0]:fg_end[0], \
        #            fg_start[1]:fg_end[1], \
        #            fg_start[2]:fg_end[2]] = pred
        orig_affine = ds[idx]["image"].meta["original_affine"]
        # recon = nib.Nifti1Image(orig_shape.astype(np.uint16), affine=orig_affine)
        recon = nib.Nifti1Image(pred.astype(np.uint8), affine=orig_affine)
        
        # recon = nib.Nifti1Image(pred.astype(np.uint16), ds[idx]['image'].affine)
        nib.save(recon, outputs[idx])

        if not args.no_pp:
            pred = post_process(pred.astype(np.uint8))
            if args.dstype != 'invivo_iso':
                pred = smooth_label(pred.astype(np.float32), 1)
            pred[pred > 0] = 255
            pred = pred.astype(np.uint8)
            recon = nib.Nifti1Image(pred, affine=orig_affine)
            nib.save(recon, outputs[idx].split('.nii')[0]+'.pp.nii.gz')

        if args.strip:
            nii = nib.load(img.meta['filename_or_obj'])
            data = nii.get_fdata()
            data[pred == 0] = 0
            recon = nib.Nifti1Image(data, affine=orig_affine)
            nib.save(recon, stripped[idx])


if __name__ == '__main__':
    main()
