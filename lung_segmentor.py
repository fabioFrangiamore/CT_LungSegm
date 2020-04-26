import numpy as np 
import pandas as pd 
import os
import json
import SimpleITK as sitk 
import registration_utils as reg_utils
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import nibabel as nib
import shutil


class CtLungSegmentor():

    def __init__(self, config_file = 'Config/config.json',
                resized_data_dir = 'resized_data',
                output_file_dir = 'Output',
                lungs_proj_file_path='Projections/lungproj_xyzb_130_py.txt',
                temporary_folder = 'tmp'):

        self.config_file = config_file 
        self.resized_data_dir = resized_data_dir
        self.output_file_dir = output_file_dir
        self.lungs_proj_file_path = lungs_proj_file_path
        self.temporary_folder = temporary_folder

    def process_file(self, file_path):

        os.makedirs(self.temporary_folder, exist_ok=True)

        with open(self.config_file, 'r') as file_conf:
            config = json.load(file_conf)

        num_nearest = config['num_nearest']
        resz = config['slice_resize']
        pos_val = config['positive_value']

        xyz_bounds, lung_projs = reg_utils.retrieve_projections(self.lungs_proj_file_path)

        img, voxel_dimensions, affine, shape0, origin, spacing = reg_utils.return_nii_data(file_path)

        img_originale = sitk.ReadImage(file_path)

        lungs = reg_utils.catch_lungs(img, voxel_dimensions)
        print(lungs.astype(int))

        projections, bounds = reg_utils.calculate_lung_projections(lungs, voxel_dimensions)

        distances = distance_matrix(projections, lung_projs).flatten()
        idx = np.argsort(distances)
        print(distances)
        print(idx)
        print(distances[idx[0]])

        ids = idx[0:num_nearest] + 1
        fixed = reg_utils.make_uint8(img[::resz, ::resz, :]).copy()
        mean_mask = (fixed * 0).astype(np.float32)

        mask_tot = []
        for j in range(num_nearest):
            path_img = os.path.join(self.resized_data_dir, 'id%03i_img.npz' % ids[j])
            data = np.load(path_img)
            moving = data[data.files[0]]
            print(path_img)

            path_msk = os.path.join(self.resized_data_dir, 'id%03i_msk.npz' % ids[j])
            data = np.load(path_msk)
            mask = data[data.files[0]]

            moving_img_path, moving_mask_path, fixed_file_path = reg_utils.save_nii_format(fixed, moving, mask, self.temporary_folder, path_img, path_msk, file_path)

            mask_out = reg_utils.affine_registration(fixed_file_path, moving_img_path, moving_mask_path, self.temporary_folder)

            mask_out_img = nib.load(mask_out)
            mask_out_img = mask_out_img.get_fdata()
            mean_mask += mask_out_img.astype(np.float32) / pos_val / num_nearest
        mean_mask= np.transpose(mean_mask, (2,0,1))
        mean_mask = np.swapaxes(mean_mask, 1, 2)
        mean_mask[mean_mask>=0.4] = 1
        mean_mask[mean_mask<0.4] = 0
        mean_mask = reg_utils.imresize(mean_mask, (93, 630, 630))
        img_originale_array = sitk.GetArrayFromImage(img_originale)
        
        img_originale_array[mean_mask==0] = 0

        mean_mask = sitk.GetImageFromArray(mean_mask)
        mean_mask.SetOrigin(origin)
        mean_mask.SetSpacing(spacing)

        img_originale_lung = sitk.GetImageFromArray(img_originale_array)
        os.makedirs('Output', exist_ok=True)
        os.makedirs('Output_lung', exist_ok=True)
        file_name_out = file_path.split('/')[1].split('.')[0]
        sitk.WriteImage(img_originale_lung,'Output_lung/{}_mask_output.nii.gz'.format(file_name_out))
        sitk.WriteImage(mean_mask, 'Output/{}_mask_output.nii.gz'.format(file_name_out))
        shutil.rmtree(self.temporary_folder)



   

        
