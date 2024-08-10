
import nibabel as nib
import numpy as np
import sys
from einops import rearrange
import torch

def sphenc(fn,ofn):
    """
    Function for generating spherical coordinates. 

    Takes in a NIfTI file to attain the data size. The Euclidean coordinates are normalized between [-1,1], with zero being in the center. 
    The individual (x,y,z) coordinates are converted to spherical coordinates.

    Parameters
    ----------
    fn : str
        Input file name. To be used for determining data size.
    ofn : str
        Output spherical coordinate encoding file name.

    """

    print("Generating spherical coordinate positional encoding for", fn)
    nii = nib.load(fn)
    X, Y, Z = nii.shape
    x = np.linspace(-1,1,X)
    y = np.linspace(-1,1,Y)
    z = np.linspace(-1,1,Z)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    
    r = np.sqrt(xv**2 + yv**2 + zv**2)
    theta = np.arctan((yv+np.finfo(float).eps)/(xv+np.finfo(float).eps))
    phi = np.arctan(np.sqrt(xv**2+yv**2)/(zv+np.finfo(float).eps))
    
    r = np.expand_dims(r, 3)
    theta = np.expand_dims(theta, 3)
    phi = np.expand_dims(phi, 3)
    
    spherecoords = np.concatenate([r, theta, phi], 3)
    recon = nib.Nifti1Image(spherecoords.astype(np.float32), nii.affine)
    nib.save(recon, ofn)

def norm_coords(fn, ofn, low=0, high=10):
    """
    Function for generating normalized coordinates. 

    Takes in a NIfTI file to attain the data size. The Euclidean coordinates are normalized between `low` and `high`. 

    Parameters
    ----------
    fn : str
        Input file name. To be used for determining data size.
    ofn : str
        Output file name containing normalized coordinates.
        
    """


    print("Generating normalized coordinates for", fn)
    nii = nib.load(fn)
    X, Y, Z = nii.shape[:3]
    # affine = nii.affine[:]

    x = np.linspace(low,high,X, dtype=np.float32)
    y = np.linspace(low,high,Y, dtype=np.float32)
    z = np.linspace(low,high,Z, dtype=np.float32)

    coords = np.meshgrid(x, y, z, indexing='ij')
    xv, yv, zv = [np.expand_dims(c, 3) for c in coords]
    normcoords = np.concatenate([xv, yv, zv], 3)
    
    recon = nib.Nifti1Image(normcoords, nii.affine)
    nib.save(recon, ofn)

def norm_sinenc(norm_coords, patch_size, D=48):
    """
    Function for computing sinusoidal functions for global absolute positional encodings. 

    Takes in a data matrix containing normalized Euclidean coordinates and returns a matrix with sinusoidal functions.

    Parameters
    ----------
    norm_coords : torch.array
        Multi-dimensional torch array with normalized Euclidan coordinates.
    patch_size : str
        Patch size or the shape the `norm_coords`.
    D : int
        Number of elements in the positional encoding vector.
        
    """

    dtype = torch.float32
    base = 10

    size = patch_size[:1]+ patch_size[2:] + (D,) 
    if len(patch_size) == 5:        
        dim = 4
        eionops_str = 'b x y z c -> b c x y z'
    elif len(patch_size) == 4:        
        dim = 3
        eionops_str = 'b x z c -> b c x z'
    else:
        print("Invalid number of dimensions.")
        sys.exit(2)
        
    PE = torch.zeros(size, dtype=dtype)
    
    # partially vectorize computations

    # separate the x,y,z norm coords
    x, y, z = (norm_coords[:,0], norm_coords[:,1], norm_coords[:,2])

    # loop through the 6 sets of 8
    ## even x sine 
    d = 0
    for i in range(0,int(D/3),2):
        PE[..., i] = torch.sin(x/(base**(6*d/D))).type(dtype)
        d += 1

    ## odd x cosine
    d = 0
    for i in range(1,int(D/3),2):
        PE[..., i] = torch.cos(x/(base**(6*d/D))).type(dtype)
        d += 1

    ## even y sine 
    d = 0
    for i in range(int(D/3),int(2*(D/3)),2):
        PE[...,i] = torch.sin(y/(base**(6*d/D))).type(dtype)
        d += 1
    
    ## odd y cosine
    d = 0
    for i in range(int(D/3)+1,int(2*(D/3)),2):
        PE[..., i] = torch.cos(y/(base**(6*d/D))).type(dtype)
        d += 1

    ## even z sine 
    d = 0
    for i in range(int(2*(D/3)),D,2):
        PE[..., i] = torch.sin(z/(base**(6*d/D))).type(dtype)
        d += 1

    ## odd z cosine
    d = 0
    for i in range(int(2*(D/3))+1,D,2):
        PE[...,i] = torch.cos(z/(base**(6*d/D))).type(dtype)
        d += 1

    PE = rearrange(PE, eionops_str)
    
    return PE