#!/usr/bin/env python3

from __future__ import unicode_literals, print_function
import os
import sys
from scipy.ndimage import binary_fill_holes
from skimage.morphology import label as sklabel
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import nibabel as nib
import argparse
from skimage.filters import gaussian


def post_process(data):
    """
    Function for closing holes and selecting the largest connected component. 
    Returns a uint8 binary mask.

    Parameters
    ----------
    fn : np.array
        Data matrix containing only labels 0 and 1.
        
    """
    
    # fill bg
    data = binary_fill_holes(data)
    # get largest connected component
    ccs = sklabel(data)
    data = ccs == np.argmax(np.bincount(ccs.flat)[1:])+1
    data = data.astype(np.uint8)
    
    return data

def smooth_label(data, sigma):
    """
    Function for smoothing binary masks.

    Parameters
    ----------
    fn : np.array
        Data matrix containing only labels 0 and 1.
    sigma : int
        Sigma for Gaussian smoothing (in voxels).
        
    """
    smoothed = gaussian(data, sigma=sigma)
    smoothed[smoothed < 0.5] = 0
    smoothed[smoothed > 0] = 1

    return smoothed

def parser():
    parser = argparse.ArgumentParser(description='Performs post processing.')
    parser.add_argument('-i', help="Input file name.", required=False)
    parser.add_argument('-o', help="Output file name.", required=True)
    return parser


def main():
    args = parser().parse_args()
    
    if not os.path.exists(os.path.dirname(args.o)):
        print('Output folder does not exist. Please make sure the output folder exists.')
        sys.exit(2)

    nii = nib.load(args.i)
    data = np.array(nii.dataobj).astype(np.uint8)
    # fill bg
    data = binary_fill_holes(data)
    # get largest connected component
    ccs = sklabel(data)
    data = ccs == np.argmax(np.bincount(ccs.flat)[1:])+1
    data = data.astype(np.uint8)
    recon = nib.Nifti1Image(data, affine=nii.affine)
    
    nib.save(recon, args.o)

if __name__ == '__main__':
    main()
