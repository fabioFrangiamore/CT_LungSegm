import os
import numpy as np
import nibabel as nib
from skimage import measure, transform
from scipy.ndimage.morphology import binary_dilation
import pandas as pd
from scipy.spatial import distance_matrix
import SimpleITK as sitk
import subprocess

def retrieve_projections(file_name):
    df = pd.read_csv(file_name, header = None)
    data = df.get_values()

    xyz_bounds = data[:, 300:306]
    lung_projs = data[:, 0:300]

    return xyz_bounds, lung_projs


def imresize(m, new_shape, order=1, mode='constant'):
    data_type = m.dtype

    multiplier = np.max(np.abs(m)) * 2
    m = m.astype(np.float32) / multiplier
    m = transform.resize(m, new_shape, order=order, mode=mode)
    m = m * multiplier

    return m.astype(dtype=data_type)

def imresize_nii(m, new_shape, order=1, mode='constant'):
    m = sitk.GetImageFromArray(m)
    print("Immagine caricata")
    im = sitk.Expand(m, [2,1,1] * 3, sitk.sitkNearestNeighbor)
    print("Immagine ridimensionata")
    origin = m.GetOrigin()
    spacing = m.GetSpacing()
    m = sitk.GetArrayFromImage(im)
    data_type = m.dtype
    


    return m.astype(dtype=data_type), origin, spacing

def return_nii_data(img):
    img_ = sitk.ReadImage(img)
    img = nib.load(img)
    
    affine = img.affine 
    img = img.get_fdata() + 1024
    img = np.swapaxes(img, 0, 1)
    img = img[::-1, :, :]

    #print(img.shape)
    voxel_dimensions = np.abs(np.diag(affine[:3, :3]))
    shape0 = img.shape
    origin = img_.GetOrigin()
    spacing = img_.GetSpacing()
    if voxel_dimensions[2] < 1.5:
        img = img[:, :, ::2].copy()
        voxel_dimensions[2] *= 2
    elif voxel_dimensions[2] > 3:
        new_size = (img.shape[0], img.shape[1], img.shape[2] * 2)
        img, origin, spacing = imresize_nii(img, new_size, order=0)
        voxel_dimensions[2] /= 2


    return img, voxel_dimensions, affine, shape0, origin, spacing

def catch_lungs(im3, voxel_dimensions):
    d3 = int(round(2.5 / voxel_dimensions[2]))
    sml = im3[::4, ::4, ::d3]
    sbw = sml > 700
    se = np.ones((3, 3, 3), dtype=bool)
    sbw = binary_dilation(sbw, se)

    sbw = np.invert(sbw).astype(int)
    lbl = measure.label(sbw)
    num_labels = np.max(lbl)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = lbl.shape
                l = lbl[i * (s[0] - 1), j * (s[1] - 1), k * (s[2] - 1)]
                lbl[lbl == l] = 0

    for i in range(num_labels):
        if np.sum(lbl == i) < 100:
            lbl[lbl == i] = 0

    lung = lbl[::2, ::2, ::2] > 0
    return lung

def trim_projection(projection):
    cumsum = np.cumsum(projection.astype(np.float) / np.sum(projection))

    d = 3
    i1 = np.where(cumsum > 0.01)[0][0] - d
    i2 = np.where(cumsum < 0.99)[0][-1] + d
    bounds = (max((0, i1)), min((len(projection), i2 + 2)))
    projection = projection[bounds[0]:bounds[1]]

    projection = np.asarray([projection])
    projection = imresize(projection, (1, 100))
    projection = projection.astype(np.float32) / projection.sum()

    return projection, np.asarray(bounds)



def calculate_lung_projections(lung, voxel_dimensions):
    x_proj = np.sum(lung, axis=(0, 2)).flatten()
    y_proj = np.sum(lung, axis=(1, 2)).flatten()
    z_proj = np.sum(lung, axis=(0, 1)).flatten()

    x_proj, xb = trim_projection(x_proj)
    y_proj, yb = trim_projection(y_proj)
    z_proj, zb = trim_projection(z_proj)

    projections = np.append(x_proj, (y_proj, z_proj))
    projections[projections < 0] = 0
    d3 = int(round(2.5 / voxel_dimensions[2]))
    mlt = (8, 8, (2 * d3))

    xyz_bounds = np.append(xb * mlt[0], (yb * mlt[1], zb * mlt[2]))

    projections = np.asarray([projections])
    return projections, xyz_bounds


def make_uint8(im):
    im = im * 0.17
    im[im < 0] = 0
    im[im > 255] = 255
    im = im.astype(np.uint8)

    return im

def save_nii_format(fixed, moving_img, mask_img, tmp_folder, path_img, path_msk, fixed_path):
    name_img = path_img.split('/')[1].split('.')[0]
    name_mask = path_msk.split('/')[1].split('.')[0]
    name_fixed = fixed_path.split('/')[1].split('.')[0]

    os.makedirs(tmp_folder + '/' + name_img,   exist_ok=True)
    os.makedirs(tmp_folder + '/' + name_mask,  exist_ok=True)
    os.makedirs(tmp_folder + '/' + name_fixed, exist_ok=True)

    img = np.transpose(moving_img, (2,0,1))[:,::-1,:]
    msk = np.transpose(mask_img, (2,0,1))[:,::-1,:]
    fxd = np.transpose(fixed, (2,0,1))[:,::-1,:]
    img = sitk.GetImageFromArray(img)
    msk = sitk.GetImageFromArray(msk)
    fixed = sitk.GetImageFromArray(fxd)
    img_file_path = tmp_folder + '/' + name_img + '/' + name_img + '.nii.gz'
    msk_file_path = tmp_folder + '/' + name_mask + '/' + name_mask + '.nii.gz'
    fixed_file_path = tmp_folder + '/' + name_fixed + '/' + name_fixed + '_resized.nii.gz'
    sitk.WriteImage(img, img_file_path)
    sitk.WriteImage(msk, msk_file_path)
    sitk.WriteImage(fixed, fixed_file_path)

    return img_file_path, msk_file_path, fixed_file_path

def affine_registration(fixed_path, moving_img_path, moving_mask_path, tmp_folder):
    
    trasformation = 'affine'
    name_img = fixed_path.split('/')[1].split('.')[0]
    moving_name = moving_img_path.split('/')[1].split('.')[0]
    name = moving_name.split('_')[0]
    print("Confronto con la figura {}".format(moving_name))
    
    my_txt = os.path.join(tmp_folder + '/' + moving_name + "/" + moving_name + '_img_affine.txt')
    txt = open(my_txt, "w")
    txt.write("[GLOBAL] \n\nfixed = " + fixed_path +"\nmoving = " + moving_img_path + "\nimg_out = " + tmp_folder + "/" + moving_name + "/" + moving_name +
            "_affine.nii.gz\nxform_out = " + tmp_folder + "/" + moving_name + "/" + moving_name + "_affine_img_coef.txt\n\n[STAGE] \nxform=" + trasformation +  "\noptim = rsg \nmax_its=30 \nres=4 4 2")
    txt.close()
    command = ['plastimatch', 'register' ,  my_txt]
    registration = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, errors = registration.communicate()
    #print ([registration.returncode, errors, output])


    my_txt = os.path.join(tmp_folder + '/' + name + "_msk" + "/" + name + '_msk_affine.txt')
    txt = open(my_txt, "w")
    txt.write("[GLOBAL] \n\nfixed = " + tmp_folder + "/" + moving_name + "/" + moving_name +
            "_affine.nii.gz" +"\nmoving = " + moving_mask_path + "\nimg_out = " + tmp_folder + '/' + name + "_msk" + "/" + name + "_msk_affine.nii.gz\nxform_in = " + 
            tmp_folder + "/" + moving_name + "/" + moving_name + "_affine_img_coef.txt")
    txt.close()
    command = ['plastimatch', 'register' ,  my_txt]
    registration = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, errors = registration.communicate()
    #print ([registration.returncode, errors, output])

    my_txt = os.path.join(tmp_folder + '/' + moving_name + "/" + moving_name + '_img_spline.txt')
    txt = open(my_txt, "w")
    txt.write("[GLOBAL] \n\nfixed = " + fixed_path +"\nmoving = " +  tmp_folder + "/" + moving_name + "/" + moving_name +
            "_affine.nii.gz" + "\nimg_out = " + tmp_folder + "/" + moving_name + "/" + moving_name +
            "_spline.nii.gz\nxform_out = " + tmp_folder + "/" + moving_name + "/" + moving_name + "_bspline_img_coef.txt\n\n[STAGE]\n xform = bspline\noptim=lbfgsb\nmax_its=400\nres= 4 4 2\ngrid_spac= 1 1 1\nregularization_lambda=0.1")
    txt.close()
    command = ['plastimatch', 'register' ,  my_txt]
    registration = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, errors = registration.communicate()
    #print ([registration.returncode, errors, output])

    my_txt = os.path.join(tmp_folder + '/' + name + "_msk" + "/" + name + '_msk_spline.txt')
    txt = open(my_txt, "w")
    txt.write("[GLOBAL] \n\nfixed =" + tmp_folder + "/" + moving_name + "/" + moving_name +
            "_spline.nii.gz" +"\nmoving = " + tmp_folder + '/' + name + "_msk" + "/" + name + "_msk_affine.nii.gz" + "\nimg_out = " + tmp_folder + '/' + name + "_msk" + "/" + name + "_msk_spline.nii.gz\nxform_in = " +
             tmp_folder + "/" + moving_name + "/" + moving_name + "_bspline_img_coef.txt"+ "\n\n[STAGE]\n xform = bspline\noptim=lbfgsb\nmax_its=0\nres= 4 4 2\ngrid_spac= 1 1 1\nregularization_lambda=0.1")
    txt.close()
    command = ['plastimatch', 'register' ,  my_txt]
    registration = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, errors = registration.communicate()
    #print ([registration.returncode, errors, output])
    mask_out = tmp_folder + '/' + name + "_msk" + "/" + name + "_msk_spline.nii.gz"


    return mask_out

