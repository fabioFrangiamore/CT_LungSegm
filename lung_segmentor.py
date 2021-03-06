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
                output_file_dir_mask = 'Output_Mask',
                output_file_lung = 'Output_lung',
                lungs_proj_file_path='Projections/lungproj_xyzb_130_py.txt',
                temporary_folder = 'tmp'):

        self.config_file = config_file 
        self.resized_data_dir = resized_data_dir
        self.output_file_dir_mask = output_file_dir_mask
        self.output_file_lung = output_file_lung
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

        img, voxel_dimensions, affine, shape0, origin, spacing, direction = reg_utils.return_nii_data(file_path)

        img_originale = sitk.ReadImage(file_path)

        lungs = reg_utils.catch_lungs(img, voxel_dimensions)
        #print(lungs.astype(int))

        projections, bounds = reg_utils.calculate_lung_projections(lungs, voxel_dimensions)

        distances = distance_matrix(projections, lung_projs).flatten()
        idx = np.argsort(distances)
        #print(distances)
        #print(idx)
        #print(distances[idx[0]])

        ids = idx[0:num_nearest] + 1
        fixed = reg_utils.make_uint8(img[::resz, ::resz, :]).copy()
        mean_mask = (fixed * 0).astype(np.float32)

        mask_tot = []

        for j in range(num_nearest):
            path_img = os.path.join(self.resized_data_dir, 'id%03i_img.npz' % ids[j])
            data = np.load(path_img)
            moving = data[data.files[0]]
            

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
        mean_mask[mean_mask>=0.6] = 1
        mean_mask[mean_mask<0.6] = 0
        print("Confronto terminato\n\nridimensiono l'immagine")
        mean_mask = reg_utils.imresize(mean_mask, (shape0[2], shape0[0], shape0[1]))
        img_originale_array = sitk.GetArrayFromImage(img_originale)
        
        img_originale_array[mean_mask==0] = 0

        mean_mask = sitk.GetImageFromArray(mean_mask)
        mean_mask.SetOrigin(origin)
        mean_mask.SetSpacing(spacing)

        print(spacing)
        img_originale_lung = sitk.GetImageFromArray(img_originale_array)
        img_originale_lung.SetSpacing(spacing)
        img_originale_lung.SetDirection(direction)
        os.makedirs(self.output_file_dir_mask, exist_ok=True)
        os.makedirs(self.output_file_lung, exist_ok=True)
        file_name_out = file_path.split('/')[1].split('.')[0]
        print("Creazione immagini finali")
        sitk.WriteImage(img_originale_lung, '{}/{}_lung_output.nii.gz'.format(self.output_file_lung, file_name_out))
        sitk.WriteImage(mean_mask,  '{}/{}_mask_output.nii.gz'.format(self.output_file_dir_mask, file_name_out))
        shutil.rmtree(self.temporary_folder)


    def process_dir(self, path):
        print(path)
        file_ending = ('.nii.gz', '.nii')
        files = os.listdir(path)

        for file in files:
            if file.endswith(file_ending):
                print("Segmentazione file {}".format(file))
                self.process_file(os.path.join(path + '/' + file))
                print("Segmentazione terminata per il file {}".format(file))



   

        
