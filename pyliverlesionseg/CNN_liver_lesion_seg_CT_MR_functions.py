"""

This script contains all functions for pre-processing CT and MR images for liver and lesion segmentation, 
predicting liver segmentation through CNN, post-processing the CNN output, and the main function for predicting liver and lesions using CNN.

"""

import os
import time
from datetime import date, datetime
import shutil
import logging
from glob import glob
from fnmatch import fnmatch
import pydicom
import csv
import h5py

import numpy as np
import nibabel as nib

import logging
from logging.handlers import TimedRotatingFileHandler

from pathlib import Path

import pymirc
import pymirc.fileio     as pymf
import pymirc.image_operations as pi

import pyliverlesionseg
import pyliverlesionseg.sampling as sampling
from pyliverlesionseg.general import DeepVoxNet
from pyliverlesionseg.architectures.unet_generalized import create_unet_like_model

from scipy import ndimage
from scipy.ndimage.filters import median_filter
from scipy.ndimage import label, labeled_comprehension, find_objects, binary_erosion, binary_opening
from scipy.signal  import argrelextrema, convolve

#################################################### functions #####################################################################

def setup_logger(name, log_path, level = logging.INFO, formatter = None, mode = 'a'):
  """ wrapper function to setup a file logger with some usefule properties (format, file replacement ...)
  """

  # create log file if it does not exist
  if not log_path.exists():
    log_path.touch() 

  if formatter is None:
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt = '%m/%d/%Y %I:%M:%S %p')

  #handler = logging.FileHandler(log_file, mode = mode)        
  handler = TimedRotatingFileHandler(filename = log_path, when = 'D', interval = 7, backupCount = 4, 
                                     encoding = 'utf-8', delay = False)

  handler.setFormatter(formatter)
 
  logger = logging.getLogger(name)
  logger.setLevel(level)
  logger.addHandler(handler)

  return logger

#-------------------------------------------------------------------------

def unet_liver_predict(ct_prepro_list,
                 segment_size,
                 ct_path,
                 unet_path,
                 out_path,
                 weight_decay    = 1e-5,
                 center_sampling = True,
                 save_output     = False):
                 
   """
   the predictor for the trained U-net model
   
   Parameters 
   -----------
   
   ct_prepro_list: a list of 3d numpy arrays which contain the preprocessed CT volumes.
   
   unet_path: the path where the trained U-net model is saved.
   
   segment_size: the output segment size of the trained U-net model.
   
   weight_decay: the weight of the regularizer.
   
   center_sampling: determine if center sampling is used for the input CT images. If True, center sampling is used.
   
   save_output: if true, the output from the U-net model is saved in the out_path.
   
   Return
   ------------
   
   the list of 3d numpy arrays which contain the possibility map predicted by the U-net model.
   
   """
   
   
   
   start_time_1 = time.time()
   print("*********************************************************")
   print("Retrieving previous U-net " + unet_path) 
   
   ct_files = glob(ct_path)
   ct_files.sort()

   model = create_unet_like_model(
                        number_input_features=1,
                        subsample_factors_per_pathway=[
                            (1, 1, 1),
                            (3, 3, 3),
                            (9, 9, 9),
                            (27, 27, 27)
                        ],
                        kernel_sizes_per_pathway=[
                            [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
                            [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
                            [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
                            [[(3, 3, 3), (3, 3, 3)], []]
                        ],
                        number_features_per_pathway=[
                            [[20, 40], [40, 20]],
                            [[40, 80], [80, 40]],
                            [[80, 160], [160, 80]],
                            [[160, 160], []]
                        ],
                        output_size=segment_size,
                        padding='same',
                        upsampling='linear',
                        mask_output=False,
                        l2_reg=weight_decay or 0.0
                    )
                    
   model.load_weights(unet_path)   
   input_segmentsize = model.input_shape[1:4]        
   deepVoxNet = DeepVoxNet(model, center_sampling=center_sampling)

   # Testing data loaders
   full_testing_inputLoader = sampling.ImageLoader(ct_prepro_list)

   # X creator
   full_testing_x_creator = sampling.Concat([
             sampling.ExtractSegment3D2(full_testing_inputLoader, input_segmentsize)
                     ])
                    
   # Testing sampler
   if center_sampling:
      full_testing_sampler = sampling.ForcedUniformCenterSampler(full_testing_inputLoader)
   else:
      full_testing_sampler = sampling.ForcedUniformCoordinateSampler(full_testing_inputLoader)

   dummy = [os.path.basename(p[:-1]) for p in ct_files]
   #dummy = [p.replace("orig", "pred_median") for p in dummy]
   dummy = [p + "_pred_median" for p in dummy]
   dummy = [os.path.join(out_path, p) for p in dummy]
   test_output_paths = dummy
   cnn_pred_median_list = []
   for subject_id, test_output_path in enumerate(test_output_paths):
       print('segmenting ' + os.path.basename(ct_files[subject_id][:-1]) +
             ' ==> ' + os.path.basename(test_output_path))
       if not save_output:
          test_output_path = None
       cnn_pred_median = deepVoxNet.predict(
                           x_creator=full_testing_x_creator,
                           sampler=full_testing_sampler,
                           subject_id=subject_id,
                           out_path=test_output_path,
                           verbose=True,
                           batch_size=1,
                           auto_recalibration=False,
                           stack_recalibrated=False,
                           output_layer_idx=[0],
                           include_output_layer_name_in_out_path=False,
                           auto_return_first_only=True
                         )
       cnn_pred_median_list.append(cnn_pred_median)

   print("The total time for CNN liver prediction is {:.2f} s".format(time.time() - start_time_1))
   return cnn_pred_median_list

def unet_lesion_predict(ct_prepro_list,
                 segment_size,
                 ct_path,
                 unet_path,
                 out_path,
                 weight_decay    = 1e-5,
                 center_sampling = True,
                 save_output     = False):
                 
   """
   the predictor for the trained U-net model
   
   Parameters 
   -----------
   
   ct_prepro_list: a list of 3d numpy arrays which contain the preprocessed CT volumes.
   
   unet_path: the path where the trained U-net model is saved.
   
   segment_size: the output segment size of the trained U-net model.
   
   weight_decay: the weight of the regularizer.
   
   center_sampling: determine if center sampling is used for the input CT images. If True, center sampling is used.
   
   save_output: if true, the output from the U-net model is saved in the out_path.
   
   Return
   ------------
   
   the list of 3d numpy arrays which contain the possibility map predicted by the U-net model.
   
   """
   
   
   
   start_time_1 = time.time()
   print("*********************************************************")
   print("Retrieving previous U-net " + unet_path) 
   
   ct_files = glob(ct_path)
   ct_files.sort()

   model = create_unet_like_model(
       number_input_features=1,
       subsample_factors_per_pathway=[
           (1, 1, 1),
           (2, 2, 2),
           (4, 4, 4),
           (8, 8, 8)
       ],
       kernel_sizes_per_pathway=[
           [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
           [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
           [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
           [[(3, 3, 1), (3, 3, 3)], []]
       ],
       number_features_per_pathway=[
           [[20, 40], [40, 20]],
           [[40, 80], [80, 40]],
           [[80, 160], [160, 80]],
           [[160, 160], []]
       ],
       output_size=segment_size,
       padding='valid',
       upsampling='copy',
       mask_output=False,
       residual_connections=True,
       l2_reg=weight_decay or 0.0
       )
       
   model.load_weights(unet_path)   
   input_segmentsize = model.input_shape[1:4]        
   deepVoxNet = DeepVoxNet(model, center_sampling=center_sampling)

   # Testing data loaders
   full_testing_inputLoader = sampling.ImageLoader(ct_prepro_list)

   # X creator
   full_testing_x_creator = sampling.Concat([
             sampling.ExtractSegment3D2(full_testing_inputLoader, input_segmentsize)
                     ])
                    
   # Testing sampler
   if center_sampling:
      full_testing_sampler = sampling.ForcedUniformCenterSampler(full_testing_inputLoader)
   else:
      full_testing_sampler = sampling.ForcedUniformCoordinateSampler(full_testing_inputLoader)

   dummy = [os.path.basename(p[:-1]) for p in ct_files]
   dummy = [p + "_pred_median" for p in dummy]
   dummy = [os.path.join(out_path, p) for p in dummy]
   test_output_paths = dummy
   cnn_pred_median_list = []
   for subject_id, test_output_path in enumerate(test_output_paths):
       print('segmenting ' + os.path.basename(ct_files[subject_id][:-1]) +
             ' ==> ' + os.path.basename(test_output_path))
       if not save_output:
          test_output_path = None
       cnn_pred_median = deepVoxNet.predict(
                           x_creator=full_testing_x_creator,
                           sampler=full_testing_sampler,
                           subject_id=subject_id,
                           out_path=test_output_path,
                           verbose=True,
                           batch_size=1,
                           auto_recalibration=False,
                           stack_recalibrated=False,
                           output_layer_idx=[0],
                           include_output_layer_name_in_out_path=False,
                           auto_return_first_only=True
                         )
       cnn_pred_median_list.append(cnn_pred_median)

   print("The total time for CNN lesion prediction is {:.2f} s".format(time.time() - start_time_1))
   return cnn_pred_median_list

def crop_ct_image(ct_vol,              #3d numpy array in LPS orientation
                  voxsize_ct,
                  size_mm_max          = [None, None, None],
                  tissue_th            = -150,
                  lung_th              = -600,
                  bone_th              = 500,
                  bone_hip_ratio       = 0.7, 
                  liver_height_max     = 400,
                  liver_dome_margin_mm = 70.,
                  hip_margin_mm        = 0.):
   """
   crop the ct image so that the cropped image contains the whole abdomen in each slice. 
   The slices of the cropped image should start from the lung and end at the hip.
   
   Parameters 
   -----------
   
   ct_vol: 3d numpy array in LPS orientation which contains the image values.
   
   voxsize_ct: the CT voxel size. 
   
   size_mm_max: a list with three elements, the maximum size (mm) of the cropped image in three dimensions. (default: [None, None, None])
   
   tissue_th: the threshould for the soft tissues and bones, air should be excluded by this threshold. (default: -150)
   
   lung_th: the threshould for the air inside the lung and from the background, soft tissues and bones should be excluded through this threshold. (default: -600)
   
   bone_th: the threshould for bones, soft tissues and air should be excluded by this threshould. (default: 500)
   
   bone_hip_ratio: the ratio used to set the threshold for the values which may belong to hip. (default: 0.7)
   
   liver_height_max: the maximum size of liver in z direnction. (default: 400)
   
   liver_dome_margin_mm: the margin left for the liver deme. (default: 70)
   
   hip_margin_mm: the margin left for the hip. (default: 0)
   
   
   Return
   ------------
   
   The bounding box used to crop the input CT image
   
   """

   xsize_mm_max = size_mm_max[0]
   ysize_mm_max = size_mm_max[1]
   zsize_mm_max = size_mm_max[2]
   
   ###############################################################################################################
   #binarize the image and find the largest connected region, which is the abodomen. 
   #Crop the image in x and y direction according to the bounding box generated from the largest connected region
   ###############################################################################################################

   # calculate the intensity histogram to find the soft tissue peak
   #histo = np.histogram(ct_vol.flatten(),50)
   #if tissue_th is None:
    # use 85% of the soft tissue peak as threshold
   # tissue_th = histo[1][int(0.85*argrelextrema(histo[0], np.greater)[0][0])]

   # binarize image 
   bin_ct_vol = (ct_vol > tissue_th)
 
   # erode bin_ct_vol 
   bin_ct_vol = binary_erosion(bin_ct_vol)

   # find biggest connected regions
   labeled_array, nlbl = label(bin_ct_vol)

   # calculated volumes of connected region 
   labels = np.arange(1, nlbl+1)
   nvox   = labeled_comprehension(ct_vol, labeled_array, labels, len, int, -1)

   # bounding box of biggest connected region in soft tissue binary image
   bbox1 = find_objects(labeled_array == labels[np.argmax(nvox)])[0]

   ct_vol2 = np.zeros(ct_vol.shape) + ct_vol.min()
   ct_vol2[bbox1] = ct_vol[bbox1]

   ###############################################################################
   # try to find lungs as symmetrical blobs with low intensity
   # get binary "air image" - only for calibrating HUs!
   ###############################################################################
   
   # create a new image in which the top slice is set to soft tissue HU
   # in case the FOV of the image cuts the lungs, the air in the lung is connected to background
   ct_vol3 = ct_vol2.copy()
   ct_vol3[:,:,-1] = 0

   # binarize the image 
   bin_air_ct_vol = (ct_vol3 < lung_th)
   
   # perform binary opening to avoid that lung is connected to background via nasal cavities
   bin_air_ct_vol_tmp = bin_air_ct_vol[:,:,:-1]
   bin_air_ct_vol_tmp = binary_opening(bin_air_ct_vol_tmp, np.ones((5,5,5)))
   bin_air_ct_vol[:,:,:-1] = bin_air_ct_vol_tmp	 
	 
   # pad the binary air mask to ensure that the background is not separated into several parts
   bin_air_ct_vol_pad = np.pad(bin_air_ct_vol, ((3,3),(3,3),(0,0)), mode='constant', constant_values  = 1)
   
   # label the binary air image 
   labeled_air_array_pad, nlbl_air = label(bin_air_ct_vol_pad)
   labeled_air_array = labeled_air_array_pad[3:-3, 3:-3, :]
   labels_air = np.arange(1, nlbl_air+1)
   
   #calculate the number of voxels in each labeled region
   nvox_air = labeled_comprehension(ct_vol3, labeled_air_array, labels_air, len, int, -1)
   air_volumes = nvox_air * np.prod(voxsize_ct)

   #find the air mask excluding the air from the background, then sum the air mask along the x and y axises to obtain the air profile.
   #The air profile value from the main part of lung should be much higher than from other parts. The approximate liver dome slice can be found by 
   #finding the threshold of the air profile.
   air_volumes_tmp = air_volumes.copy()
   air_volumes_tmp.sort()
   air_mask_abd = np.logical_and(labeled_air_array != labels_air[air_volumes == air_volumes_tmp[-1]][0], bin_air_ct_vol == 1)
   air_abd_profile = convolve(air_mask_abd.sum(0).sum(0),np.ones(7), 'same')
   air_profile_max = np.max(air_abd_profile)
   air_profile_min = np.min(air_abd_profile)   
   Th = air_profile_min + 0.5*(air_profile_max - air_profile_min)
   range_lung = np.where(air_abd_profile > Th)[0]
   range_lung_diff = convolve(range_lung,np.array([1,-1]), 'same')
   idx_range_starts = np.where(range_lung_diff != 1)[0]
   if len(idx_range_starts) > 1:
     if np.any(air_abd_profile[range_lung[idx_range_starts[1]-1]:range_lung[idx_range_starts[1]]] < air_profile_min + 0.2*(air_profile_max - air_profile_min)):
       liver_dome_sl_1 = range_lung[idx_range_starts[1]]
     else:
       liver_dome_sl_1 = range_lung[0]
   else:
     liver_dome_sl_1 = range_lung[0]
   
   # find the x and y coordinates of the central point in the abdomen air mask
   bbox_air_abd = find_objects(air_mask_abd)[0]
   cent_air_abd_x = (bbox_air_abd[0].start + bbox_air_abd[0].stop + 1)//2
   cent_air_abd_y = (bbox_air_abd[1].start + bbox_air_abd[1].stop + 1)//2 
   
   # find the lung with the largest lower bound between two lungs, which should be lower than the liver dome slice
   possbile_lung_labels = np.array([labels_air[air_volumes == air_volumes_tmp[-2]][0], labels_air[air_volumes == air_volumes_tmp[-3]][0]])
   bbox_lung_0 = find_objects(labeled_air_array == possbile_lung_labels[0])[0]
   bbox_lung_1 = find_objects(labeled_air_array == possbile_lung_labels[1])[0]
   lower_bounds_z = [bbox_lung_0[2].start, bbox_lung_1[2].start]
   lung_label = possbile_lung_labels[np.argmax(lower_bounds_z)]
   bbox_lung  = find_objects(labeled_air_array == lung_label)[0]

   liver_dome_sl_2 = bbox_lung[2].start
   
   #take the maximum value between the two candidates of the approximate liver dome slices
   liver_dome_sl = max(liver_dome_sl_1, liver_dome_sl_2)

   ##########################################################################################
   # look at bone image to find hip
   ##########################################################################################
   
   #binarize the image to obtain the bone mask
   bin_bone_ct_vol = (ct_vol > bone_th)
   bin_bone_ct_vol = bin_bone_ct_vol.astype('int16')
	 
   #sum the bone mask along x and y axises to obtain a 1d bone profile along z axis
   #smooth the bone profile through convolution to remove noise 
   bone_profile = convolve(bin_bone_ct_vol.sum(0).sum(0),np.ones(7), 'same')[:liver_dome_sl]
   
   #find the maximum the slice where the bone profile reaches its maximum value
   slice_bone_max = np.argmax(bone_profile)
   
   #find the slice numbers of the local minimum values among the bone profile
   slice_bone_local_min = argrelextrema(bone_profile, np.less, order = 15)[0]
   
   #in case the order is too large and no local minima is found, decrease the order until at least one local minima is found
   order_tmp = 15
   while len(slice_bone_local_min) == 0:
     order_tmp = order_tmp - 3
     order_tmp = max(1, order_tmp)
     slice_bone_local_min = argrelextrema(bone_profile, np.less, order = order_tmp)[0]
   
   #find the slice with the smallest value among all local minima 
   bone_local_min = bone_profile[slice_bone_local_min]
   idx_sort_min = np.argsort(bone_local_min)

   slice_bone_min = slice_bone_local_min[idx_sort_min[0]]
     
   #find the first and second largest local maxima. If there is only one local maxima, take this maxima as the largest local maxima.
   slice_bone_local_max = argrelextrema(bone_profile, np.greater, order = 15)[0]
   order_tmp = 15
   while len(slice_bone_local_max) == 0:
     order_tmp = order_tmp - 3
     order_tmp = max(1, order_tmp)
     slice_bone_local_max = argrelextrema(bone_profile, np.greater, order = order_tmp)[0]
   bone_local_max = bone_profile[slice_bone_local_max]
   idx_sort = np.argsort(bone_local_max)
   if len(idx_sort) < 2:
     slice_second_local_max = slice_bone_local_max[idx_sort[-1]]
   else:
     slice_second_local_max = slice_bone_local_max[idx_sort[-2]]
     
   #if the slice with the minimum value is smaller than the slices with the maximum value and with the second largest local maxima,
   #truncate the bone profile from the slice with the second largest local maxima and find the slice with the minimum value in the truncated 
   #bone profile. 
   if slice_bone_min < slice_bone_max:
     if slice_bone_min < slice_second_local_max:
       bone_profile_tmp = bone_profile[slice_second_local_max:]
       slice_bone_min = slice_second_local_max + np.argmin(bone_profile_tmp)
       
   #find the slices with the local maximum values in the bone profile
   slice_bone_local_max = argrelextrema(bone_profile, np.greater, order = 5)[0]
   order_tmp = 5
   while len(slice_bone_local_max) == 0:
     order_tmp = order_tmp - 1
     order_tmp = max(1, order_tmp) 
     slice_bone_local_max = argrelextrema(bone_profile, np.greater, order = order_tmp)[0]     
   
   #The possible hip slices should have the bone profile values over 'bone_hip_factor' times the minimum bone profile value
   #The possbile hip slices should also belong to the slices with the local maximum values in the bone profile
   #possible_hip_slices = np.where(bone_profile > bone_hip_factor*bone_profile[slice_bone_min])[0]     
   possible_hip_slices = np.where(bone_profile > bone_hip_ratio*(bone_profile[slice_bone_max]-bone_profile[slice_bone_min]) + bone_profile[slice_bone_min])[0] 
   possible_slice_bone_local_max = np.intersect1d(slice_bone_local_max, possible_hip_slices)
   bone_hip_ratio_tmp = bone_hip_ratio
   while len(possible_hip_slices) == 0 or len(possible_slice_bone_local_max) == 0:
     bone_hip_ratio_tmp = bone_hip_ratio_tmp - 0.1
     bone_hip_ratio_tmp = max(bone_hip_ratio_tmp, 0.05)
     possible_hip_slices = np.where(bone_profile > bone_hip_ratio_tmp*(bone_profile[slice_bone_max]-bone_profile[slice_bone_min]) + bone_profile[slice_bone_min])[0] 
     possible_slice_bone_local_max = np.intersect1d(slice_bone_local_max, possible_hip_slices)

   #The hip slice should be closet to the slice with the minimum bone profile value among the possible hip slices which belong to local maximas
   #in the bone profile
   if len(np.where(possible_slice_bone_local_max < slice_bone_min)[0]) < 2:
     hip_slice = 0
   else:
     hip_slice = possible_slice_bone_local_max[np.where(possible_slice_bone_local_max < slice_bone_min)][-1]
   
   #the distance between the slice of liver dome and hip slice should not be over the defined maximum liver height
   if (liver_dome_sl - hip_slice) * voxsize_ct[2] > liver_height_max:
     hip_slice = liver_dome_sl - int(liver_height_max/voxsize_ct[2])

   #########################################################################################################################
   # crop the image based on the zstart and zend and then find the largest connected region (abdomen) in the cropped image #
   # try to remove as much as background and arms away                                                                     #
   #########################################################################################################################
   
   # create final bounding box
   zstart = max(0, hip_slice - int(hip_margin_mm/voxsize_ct[2]))
   zend   = min(ct_vol.shape[2], liver_dome_sl + int(liver_dome_margin_mm/voxsize_ct[2]))

   bbox = (bbox1[0],bbox1[1],slice(zstart,zend,None))

   # crop image
   ct_vol_cropped = ct_vol[bbox]

   # do another crop around the soft tissue in LP direction 
   bin_ct_vol = (ct_vol_cropped > tissue_th)

   #binary opening image
   bin_ct_vol = binary_erosion(bin_ct_vol, np.ones((5,5,5)))
   bin_ct_vol = bin_ct_vol.astype('int16')
   
   # find biggest connected regions
   labeled_array, nlbl = label(bin_ct_vol)

   # calculated volumes of connected region 
   labels = np.arange(1, nlbl+1)
   nvox   = labeled_comprehension(ct_vol_cropped, labeled_array, labels, len, int, -1)

   # bounding box of biggest connected region in soft tissue binary image
   bbox2 = find_objects(labeled_array == labels[np.argmax(nvox)])[0]

   tmp0 = slice(bbox[0].start + bbox2[0].start, bbox[0].start + bbox2[0].stop, None)
   tmp1 = slice(bbox[1].start + bbox2[1].start, bbox[1].start + bbox2[1].stop, None)

   bbox_final = (tmp0,tmp1,bbox[2])
   
   #######################################################################################
   #ensure the cropped image size does not exceed the maximum size in mm
   #######################################################################################
   
   #if the xsize_mm of the final cropped image is larger than xsize_mm_max, make sure xsize_mm = xsize_mm_max
   if xsize_mm_max is not None:
     xsize = bbox_final[0].stop - bbox_final[0].start
     if xsize * voxsize_ct[0] > xsize_mm_max: 
       offset_x_mm = xsize * voxsize_ct[0] - xsize_mm_max
       delta_x_start = np.ceil((cent_air_abd_x - bbox_final[0].start) / xsize * offset_x_mm / voxsize_ct[0]).astype('int16')
       delta_x_stop = np.ceil((bbox_final[0].stop - cent_air_abd_x) / xsize * offset_x_mm / voxsize_ct[0]).astype('int16')  
       bbox0_check = slice(bbox_final[0].start + delta_x_start, bbox_final[0].stop - delta_x_stop, None)
     else:
       bbox0_check = bbox_final[0]
   else:
     bbox0_check = bbox_final[0]
   
   #if the ysize_mm of the final cropped image is larger than ysize_mm_max, make sure ysize_mm = ysize_mm_max   
   if ysize_mm_max is not None:
     ysize = bbox_final[1].stop - bbox_final[1].start
     if ysize * voxsize_ct[1] > ysize_mm_max: 
       offset_y_mm = ysize * voxsize_ct[1] - ysize_mm_max
       delta_y_start = np.ceil((cent_air_abd_y - bbox_final[1].start) / ysize * offset_y_mm / voxsize_ct[1]).astype('int16')
       delta_y_stop = np.ceil((bbox_final[1].stop - cent_air_abd_y) / ysize * offset_y_mm / voxsize_ct[1]).astype('int16')   
       bbox1_check = slice(bbox_final[1].start + delta_y_start, bbox_final[1].stop - delta_y_stop, None)  
     else:
       bbox1_check = bbox_final[1]       
   else:
     bbox1_check = bbox_final[1]
 
   #if the zsize_mm of the final cropped image is larger than zsize_mm_max, crop the image so that zsize_mm = zsize_mm_max  
   if zsize_mm_max is not None:
     zsize = bbox_final[2].stop - bbox_final[2].start
     if zsize * voxsize_ct[2] > zsize_mm_max:
       delta_z_start = np.ceil((zsize * voxsize_ct[2] - zsize_mm_max) / voxsize_ct[2]).astype('int16')
       bbox2_check = slice(bbox_final[2].start + delta_z_start, bbox_final[2].stop, None)
     else:
       bbox2_check = bbox_final[2]
   else:
     bbox2_check = bbox_final[2]
     
   bbox_final_check = (bbox0_check, bbox1_check, bbox2_check)

   return bbox_final_check
   
def crop_mr_image(mr_vol,              #3d numpy array in LPS orientation
                  voxsize_mr,
                  size_mm_max          = [None, None, None],
                  air_abd_ratio        = 0.2,
                  tissue_ratio         = 0.02,
                  liver_height_max     = 400,
                  zstart_margin_mm        = 0.):
   """
   crop the mr image so that the cropped image contains the whole abdomen in each slice. 
   The slices of the cropped image should start from the lung and end at the hip.
   
   Parameters 
   -----------
   
   mr_vol: 3d numpy array in LPS orientation which contains the image values.
   
   voxsize_mr: the MR voxel size. 
   
   size_mm_max: a list with three elements, the maximum size (mm) of the cropped image in three dimensions. (default: [None, None, None])
   
   tissue_ratio: the ratio used to threshould the soft tissues, air should be excluded by this threshold. (default: -150)
                 Threshold = ratio * maximum_value_in_MR_vol
   
   air_abd_ratio: the ratio used to threshold the y-direnction profile of the mr image so that we can find the bounds in y direnction which seperates abdomen from air
   
   liver_height_max: the maximum size of liver in z direnmrion. (default: 400)
   
   zstart_margin_mm: the margin left for the starting of the slices in z-direction. (default: 0)
   
   
   Return
   ------------
   
   The bounding box used to crop the input CT image
   
   """

   xsize_mm_max = size_mm_max[0]
   ysize_mm_max = size_mm_max[1]
   zsize_mm_max = size_mm_max[2]

   mr_size = mr_vol.shape
   #########################################################################################################################
   # crop the image based on the zstart and zend and then find the largest connected region (abdomen) in the cropped image #
   # try to remove as much as background and arms away                                                                     #
   #########################################################################################################################
   
   # find the bounding values in y direction
   # calcualte the difference of y profile for the MR image
   y_profile = np.sum(mr_vol, (0,2))
   y_profile = convolve(y_profile, np.ones(7), 'same')
   y_prof_diff = convolve(y_profile, np.concatenate((np.ones(30),np.zeros(1),-np.ones(30))), 'same')
   
   # find the global minima and local maxima of y profile
   y_prof_diff_argmin = np.argmin(y_prof_diff)
   y_prof_diff_arg_local_max = argrelextrema(y_prof_diff, np.greater, order = 70)[0]
   order_tmp = 70
   while len(y_prof_diff_arg_local_max) <= 1 and order_tmp != 2:
     order_tmp = max(order_tmp - 10, 2)
     y_prof_diff_arg_local_max = argrelextrema(y_prof_diff, np.greater, order = order_tmp)[0]
   
   # find the maximum value in the abdominal range of y_profile_diff
   if len(y_prof_diff_arg_local_max) == 1:
     y_abd_max = y_prof_diff_arg_local_max[0]
   else: 
     y_prof_diff_local_max_sort = np.sort(y_prof_diff[y_prof_diff_arg_local_max])
     possbile_y_abd_maxs = y_prof_diff_arg_local_max[np.where(y_prof_diff[y_prof_diff_arg_local_max] > 0.5*y_prof_diff_local_max_sort[-2])]
     possbile_y_abd_maxs = possbile_y_abd_maxs[np.where(possbile_y_abd_maxs < y_prof_diff_argmin-100)]

     diff_tmp = 100
     while len(possbile_y_abd_maxs) < 1:
       diff_tmp = diff_tmp - 10
       possbile_y_abd_maxs = y_prof_diff_arg_local_max[np.where(y_prof_diff[y_prof_diff_arg_local_max] > 0.5*y_prof_diff_local_max_sort[-2])]
       possbile_y_abd_maxs = possbile_y_abd_maxs[np.where(possbile_y_abd_maxs < y_prof_diff_argmin-diff_tmp)]
     
     possbile_y_abd_maxs = possbile_y_abd_maxs[np.where(possbile_y_abd_maxs < 0.4 * mr_size[1])]
     
     # use the global maxima if possbile_y_abd_maxs is null
     if len(possbile_y_abd_maxs) == 0:
       y_abd_max = np.argmax(y_prof_diff)
       if y_abd_max >= 0.4 * mr_size[1]:
         y_abd_max = 0
     else:
       if len(possbile_y_abd_maxs) > 1 and np.all(y_prof_diff[possbile_y_abd_maxs[-2]:possbile_y_abd_maxs[-1]+1] > 0.5*y_prof_diff[possbile_y_abd_maxs[-1]]):
         y_abd_max = possbile_y_abd_maxs[-2]
       elif len(possbile_y_abd_maxs) > 1 and y_prof_diff[possbile_y_abd_maxs[-1]] < 0.5 * y_prof_diff[possbile_y_abd_maxs[-2]]:
         y_abd_max = possbile_y_abd_maxs[-2]   
       else:
         y_abd_max = possbile_y_abd_maxs[-1]

   # find the starting slice index of the abdomial region in y direction
   possible_y_abd_starts = np.array(np.where(y_prof_diff < 0.45*y_prof_diff[y_abd_max]))
   possible_y_abd_starts = possible_y_abd_starts[np.where(possible_y_abd_starts < y_abd_max)]
   Th_tmp = 0.45
   while len(possible_y_abd_starts) == 0 and Th_tmp < 1.0:
     Th_tmp = Th_tmp + 0.05
     possible_y_abd_starts = np.array(np.where(y_prof_diff < Th_tmp*y_prof_diff[y_abd_max]))
     possible_y_abd_starts = possible_y_abd_starts[np.where(possible_y_abd_starts < y_abd_max)]
   # set the starting point in y direction to zero if possible starting points still can't be found
   if len(possible_y_abd_starts) == 0:
     y_abd_start = 0
   else:
     y_abd_start = possible_y_abd_starts[-1]

   # find the possible stopping slice index of the abdominal region in y direction 
   possbile_y_abd_ends = np.array(np.where(y_prof_diff > 0.6*y_prof_diff[y_prof_diff_argmin]))
   possbile_y_abd_ends = possbile_y_abd_ends[np.where(possbile_y_abd_ends > y_prof_diff_argmin)]
   ratio_tmp = 0.6
   while len(possbile_y_abd_ends) < 1:
     ratio_tmp = ratio_tmp + 0.05
     if ratio_tmp > 1.0:   
       break
     possbile_y_abd_ends = np.array(np.where(y_prof_diff > ratio_tmp*y_prof_diff[y_prof_diff_argmin]))
     possbile_y_abd_ends = possbile_y_abd_ends[np.where(possbile_y_abd_ends > y_prof_diff_argmin)]
   
   # find the stopping slice index of the abdominal region in y direction
   if ratio_tmp <= 1.0: 
     possbile_y_abd_ends_diff = convolve(possbile_y_abd_ends, np.concatenate((np.ones(1),-np.ones(1))), 'same')
     idx_div = np.where(possbile_y_abd_ends_diff > 1)[0]
     if len(idx_div) > 1:
       possbile_y_abd_ends = possbile_y_abd_ends[idx_div[-1]:]
     if len(possbile_y_abd_ends) == 0:
       y_abd_end = y_prof_diff_argmin
     else:
       y_abd_end = possbile_y_abd_ends[0]
   else:
     y_abd_end = mr_size[1]

   slice_abd_y = slice(y_abd_start, y_abd_end, None)

   
   # create final bounding box
   zstart = max(0, mr_vol.shape[2] - int(liver_height_max/voxsize_mr[2]) - int(zstart_margin_mm/voxsize_mr[2]))
   zend   = mr_vol.shape[2]  
   slice_z = slice(zstart, zend, None)
   
   # crop image
   mr_vol_cropped = mr_vol[:,:,slice_z]
   
   # crop around the soft tissue in LP direction 
   bin_mr_vol = (mr_vol_cropped > tissue_ratio * np.max(mr_vol_cropped))
   
   #binary opening image
   bin_mr_vol = binary_erosion(bin_mr_vol, np.ones((5,5,5)))
   bin_mr_vol = bin_mr_vol.astype('int16')
   
   # find biggest connected regions
   labeled_array, nlbl = label(bin_mr_vol)

   # calculated volumes of connected region 
   labels = np.arange(1, nlbl+1)

   nvox   = labeled_comprehension(mr_vol_cropped, labeled_array, labels, len, int, -1)

   # bounding box of biggest connected region in soft tissue binary image
   bbox = find_objects(labeled_array == labels[np.argmax(nvox)])[0]

   tmp0 = slice(bbox[0].start, bbox[0].stop, None)
   tmp1 = slice(bbox[1].start, bbox[1].stop, None)
   
   slice_y = slice(max(tmp1.start, slice_abd_y.start), min(tmp1.stop, slice_abd_y.stop), None)
   
   bbox_final = (tmp0,slice_y,slice_z)
   
   #######################################################################################
   #ensure the cropped image size does not exceed the maximum size in mm
   #######################################################################################
   
   #if the xsize_mm of the final cropped image is larger than xsize_mm_max, make sure xsize_mm = xsize_mm_max
   if xsize_mm_max is not None:
     xsize = bbox_final[0].stop - bbox_final[0].start
     if xsize * voxsize_mr[0] > xsize_mm_max: 
       offset_x_mm = xsize * voxsize_mr[0] - xsize_mm_max
       delta_x_start = np.ceil(offset_x_mm / (2 * voxsize_mr[0])).astype('int16')
       delta_x_stop = np.ceil(offset_x_mm / (2 * voxsize_mr[0])).astype('int16')
       bbox0_check = slice(bbox_final[0].start + delta_x_start, bbox_final[0].stop - delta_x_stop, None)
     else:
       bbox0_check = bbox_final[0]
   else:
     bbox0_check = bbox_final[0]
   
   #if the ysize_mm of the final cropped image is larger than ysize_mm_max, make sure ysize_mm = ysize_mm_max   
   if ysize_mm_max is not None:
     ysize = bbox_final[1].stop - bbox_final[1].start
     if ysize * voxsize_mr[1] > ysize_mm_max: 
       offset_y_mm = ysize * voxsize_mr[1] - ysize_mm_max
       delta_y_start = np.ceil(offset_y_mm / (2 * voxsize_mr[1])).astype('int16')
       delta_y_stop = np.ceil(offset_y_mm / (2 * voxsize_mr[1])).astype('int16')
       bbox1_check = slice(bbox_final[1].start + delta_y_start, bbox_final[1].stop - delta_y_stop, None)  
     else:
       bbox1_check = bbox_final[1]       
   else:
     bbox1_check = bbox_final[1]
 
   #if the zsize_mm of the final cropped image is larger than zsize_mm_max, crop the image so that zsize_mm = zsize_mm_max  
   if zsize_mm_max is not None:
     zsize = bbox_final[2].stop - bbox_final[2].start
     if zsize * voxsize_mr[2] > zsize_mm_max:
       delta_z_start = np.ceil((zsize * voxsize_mr[2] - zsize_mm_max) / voxsize_mr[2]).astype('int16')
       bbox2_check = slice(bbox_final[2].start + delta_z_start, bbox_final[2].stop, None)
     else:
       bbox2_check = bbox_final[2]
   else:
     bbox2_check = bbox_final[2]
     
   bbox_final_check = (bbox0_check, bbox1_check, bbox2_check)

   return bbox_final_check
   
def cnn_liver_pred_postprocess(cnn_pred,
                         img_vol,
                         bbox,
                         Th = 0.5,
                         flag_flip = True):
                         
   """
   
   postprocess the CNN prediction through thresholding, finding the largest connected region, and padding.
   
   Parameters 
   -----------

   cnn_pred: the ouput probability map from the CNN model.
  
   img_vol: a numpy array containing the volume in LPS

   bbox: the bounding box of the cropped image, which is used for padding the cnn_pred to the original size of the CT.   
   
   Th: the threshold used for binarizing the probability map.
   
   flag_flip: if true, flip the predtion in x and y directions to convert it back to LPS.
                         
   Return
   ------------

   the postprocessed cnn prediction, which can match the orignal CT. 
   
   """                     

   start_time = time.time()
   
   #read dcm
   #ct_dcm = pymf.DicomVolume(dcm_files)
   #ct_vol = ct_dcm.get_data()

   # in case we get a 4D volume (e.g. multi echo MR, we use the first frame/echo)
   if img_vol.ndim == 4:
     img_vol = img_vol[0,:,:,:]
   
   # the CNN output needs to be in LPS orientation for post-processing of CNN liver seg but not of CNN lesion seg (mainly for padding)
   if flag_flip:
     cnn_pred = np.flip(cnn_pred, [0,1])
   
   # binarize the output probability map from CNN
   cnn_pred_bi = cnn_pred > Th
 
   # binary opening
   struct = ndimage.generate_binary_structure(3,2)
   ss = ndimage.iterate_structure(struct, 1).astype(int)
   cnn_pred_erode = ndimage.binary_erosion(cnn_pred_bi, structure=ss).astype(int)
	 
	 # find the largest connected region
   struct = ndimage.generate_binary_structure(3,1)
   cnn_lbl, nb_labels = ndimage.label(cnn_pred_erode, structure=struct)
   sizes = ndimage.sum(cnn_pred_erode, cnn_lbl, range(nb_labels + 1))
   sort_idx = np.argsort(sizes)
   cnn_liv_bi = cnn_lbl == sort_idx[-1]
	 
	 # dilate the largest connected region
   cnn_pred_dil = ndimage.binary_dilation(cnn_liv_bi, structure=ss)
   cnn_pred_dil = cnn_pred_dil.astype('int16')

   # resize the CNN output to the orignal voxel size
   zoom_factor = np.array(img_vol[bbox].shape) / np.array(cnn_pred_dil.shape)
   cnn_pred_resize = ndimage.zoom(cnn_pred_dil, zoom_factor, order = 1, prefilter = False)
   cnn_pred_resize = cnn_pred_resize > 0
   cnn_pred_resize = cnn_pred_resize.astype('int16')
   
   # pad the resized CNN output to the orignal size of CT
   cnn_pred_pad = np.zeros_like(img_vol)
   cnn_pred_pad[bbox] = cnn_pred_resize
   
   print(f'The time for postprocessing is {(time.time() - start_time):.2f} s')
   return cnn_pred_pad

def cnn_lesion_pred_postprocess(cnn_pred,
                         img_vol,
                         bbox,
                         img_resize_orig_shape,
                         Th = 0.5,
                         flag_flip = True):
                         
   """
   
   postprocess the CNN prediction through thresholding, finding the largest connected region, and padding.
   
   Parameters 
   -----------

   cnn_pred: the ouput probability map from the CNN model.
  
   img_vol: a numpy array containing the volume in LPS

   bbox: the bounding box of the cropped image, which is used for padding the cnn_pred to the original size of the CT.   
   
   img_resize_orig_shape: the shape of the resize image before cropping
   
   Th: the threshold used for binarizing the probability map.
   
   flag_flip: if true, flip the predtion in x and y directions to convert it back to LPS.
                         
   Return
   ------------

   the postprocessed cnn prediction, which can match the orignal CT. 
   
   """                     

   start_time = time.time()
   
   #read dcm
   #ct_dcm = pymf.DicomVolume(dcm_files)
   #ct_vol = ct_dcm.get_data()

   # in case we get a 4D volume (e.g. multi echo MR, we use the first frame/echo)
   if img_vol.ndim == 4:
     img_vol = img_vol[0,:,:,:]
   
   # the CNN output needs to be in LPS orientation for post-processing of CNN liver seg but not of CNN lesion seg (mainly for padding)
   if flag_flip:
     cnn_pred = np.flip(cnn_pred, [0,1])
   
   # binarize the output probability map from CNN
   cnn_pred_bi = cnn_pred > Th
 
   # pad the CNN lesion pred
   cnn_pred_pad = np.zeros(img_resize_orig_shape)
   cnn_pred_pad[bbox] = cnn_pred_bi
   
   # resize the CNN output to the orignal voxel size
   zoom_factor = np.array(img_vol.shape) / np.array(cnn_pred_pad.shape)
   cnn_pred_resize = ndimage.zoom(cnn_pred_pad, zoom_factor, order = 1, prefilter = False)
   cnn_pred_resize = cnn_pred_resize > 0
   cnn_pred_resize = cnn_pred_resize.astype('int16')

   print(f'The time for postprocessing CNN lesion seg is {(time.time() - start_time):.2f} s')
   return cnn_pred_resize
   
def move_to_failed_and_cleanup(process_dir, fail_dir, logger):
  """ move directory to failed archiv and clean up empty parent directories

  Parameters
  ----------

  process_dir ... dicom directory where processing failed

  fail_dir    ... archiv directory for failed processings

  logger      ... logger to log infos
  """
  fdir = os.path.join(fail_dir, os.path.basename(process_dir))

  # if the failed dir already exists, we have to remove it first
  if os.path.exists(fdir):
    shutil.rmtree(fdir) 

  shutil.move(process_dir, fdir)
  logger.info(f'moved input to {fdir}')

  # remove parent dir if it is empty
  parent_dir = os.path.dirname(process_dir)
  if (fnmatch(os.path.basename(parent_dir),'20??-??__Studies')) and  (len(os.listdir(parent_dir)) == 0):
    os.rmdir(parent_dir)
    logger.info(f'removed empty directory {parent_dir}')

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

def CNN_liver_lesion_seg_CT_MR_main(process_dir, liver_seg_dir, WholeLiverModel, LesionsModel, seg_liver = False, seg_lesion = False, save_nifti = False, input_nifti = False, Modality = None):

  """
  the main function for CNN liver lesions segmentation
  
  Parameters 
  -----------

  process_dir ... directory containing the dicom files of the input volume
  
  liver_seg_dir ... directory containing the dicom files of the whole liver RTstruct
                      if None, this RTstruct will be generated by the WholeLiverModel
                      
  WholeLiverModel ... name of the model for the whole liver segmentation (h5 file)
  
  LesionsModel ... name of the model for the liver lesions segmentation (h5 file)
  
  seg_liver ... boolean, default = False. Whether to segment liver using CNN.
  
  seg_lesion ... boolean, default = False. Whether to segment lesions using CNN.
  
  save_nifti ... boolean, default = False. Whether to save the CNN segmentation as nifti file
  
  input_nifti ... boolean, default = False. If true, the input image is in NIFTI format. If false, the input image is in DICOM format.
  
  Modality ... 'CT' or 'MR', default = None. When input_nifti is true, Modality has to be given.
  
  """   

  logger = setup_logger('CNN_logger', Path(process_dir).parent / 'CNN_processing.log')

  # the voxel size of the images before imported into the trained model for liver seg    
  with h5py.File(WholeLiverModel, 'r') as f_liver_model:
    target_voxsize_liver = np.array(list(f_liver_model['header/voxel_size']))
  #the voxel size of the images before imported into the trained model for lesion seg
  with h5py.File(LesionsModel, 'r') as f_lesions_model:
    target_voxsize_lesion = np.array(list(f_lesions_model['header/voxel_size']))

  output_series_liver_seg_path = os.path.dirname(process_dir)
  output_series_lesion_seg_path = os.path.dirname(process_dir)
  
  #--------------------------------------------------------------------------------------------

  target_voxsize_liver_1 = target_voxsize_liver / 2.
  target_voxsize_liver_2 = target_voxsize_liver
  
  #--------------------------------------------------------------------------------------------
  
  logger.info(f'')
  logger.info(f'======================================================')
  logger.info(f'processing: {process_dir}')
  logger.info(f'')

  # read dcm files
  if not input_nifti:
    dcm_files = sorted(glob(os.path.join(process_dir,'*.dcm')))
    
    # check whether all dicom files contain pixel data
    # if e.g. presentation state files are present (from Philips MR). the reader will fail
    # we simply ignore all dicom files that do not contain pixel data
    dcm_files = [x for x in dcm_files if 'PixelData' in pydicom.read_file(x)]
    
    # it can happen that the dicom reading fails when using untested dicom data
    # which is why we have to catch possible errors
    try:
      img_dcm = pymf.DicomVolume(dcm_files)
    except:
      # if the dicom reading fails with an error we move the dicom directory
      # to the fail dir and continue with the next data set
      logger.error(f'Failed to read dicoms from {process_dir}')
    
    ###############################################################################
    # if data reading was successfull the real processing starts here
    ###############################################################################
    
    try:
      img_vol = img_dcm.get_data()
      voxsize_img = img_dcm.voxsize
      # read image modality
      Modality = img_dcm.firstdcmheader.Modality     
      affine_lps = img_dcm.affine      
    except:
      logger.error(f'Failed to read dicoms from {process_dir}')
  else:
    img_nii = nib.load(process_dir)
    img_nii = nib.as_closest_canonical(img_nii)
    img_vol = img_nii.get_data()  
    # transform the volume to LPS
    img_vol = np.flip(img_vol, [0,1])
    voxsize_img = img_nii.header['pixdim'][1:4]
    # read the affine
    affine_lps       = img_nii.affine.copy()
    affine_lps[0,-1] = (-1 * img_nii.affine @ np.array([img_vol.shape[0]-1,0,0,1]))[0]
    affine_lps[1,-1] = (-1 * img_nii.affine @ np.array([0,img_vol.shape[1]-1,0,1]))[1]
    
    if Modality is None:
      logger.error(f'Modliaty has to be given when the input image is in NIFTI format. Modality should be CT or MR.')
      
  # in case we get a 4D volume (e.g. multi echo MR, we use the first frame/echo)
  if img_vol.ndim == 4:
    img_vol = img_vol[0,:,:,:]
  
  ###############################################################################
  # crop CT/MR images and then median filter and resize the cropped CT images
  ###############################################################################
  
  img_prepro_list = []
  start_time_0 = time.time()
  
  start_time_tmp = time.time()
  imgname = os.path.basename(process_dir)
  logger.info(f'{imgname}')
  
  # start to run CNN liver segmentation first to obtain liver mask
  # when liver_seg_dir is not None, use liver mask from the doctor
  # =================================================================================================================
  # ====================================CNN liver seg ===============================================================
  # =================================================================================================================
  if seg_liver or (seg_lesion and liver_seg_dir is None):
  
    # the img should be in LPS orientation (z axis (2 axis) should increase from feet to head) 
    if Modality == 'CT': 
      bbox = crop_ct_image(img_vol,  #3d numpy array in LPS orientation
                         voxsize_img,
                         size_mm_max = [489, 408, 408],
                         tissue_th  = -150,
                         lung_th   = -600,
                         bone_th   = 500,
                         bone_hip_ratio = 0.7, 
                         liver_height_max = 400,
                         liver_dome_margin_mm = 70.,
                         hip_margin_mm        = 20.)
    elif Modality == 'MR':
      bbox = crop_mr_image(img_vol,  #3d numpy array in LPS orientation
                         voxsize_img,
                         size_mm_max      = [489, 408, 408],
                         tissue_ratio     = 0.01,
                         liver_height_max = 400,
                         zstart_margin_mm = 30.)                          
    else:
      # if the dicom modality is not CT nor MR, we move the dicom directory
      # to the fail dir and continue with the next data set
      logger.error(f'Dicom modality {Modality} is not supported')
    
    # save the bounding box in a csv file
    with open(os.path.join(output_series_liver_seg_path, imgname + '_bbox.csv'), 'w') as csvfile:
       writer = csv.writer(csvfile) 
       writer.writerow(['x-start', 'x-stop', 'y-start', 'y-stop', 'z-start', 'z-stop'])
       writer.writerow([bbox[0].start, bbox[0].stop, bbox[1].start, bbox[1].stop, bbox[2].start, bbox[2].stop])
     
    img_vol_cropped = img_vol[bbox]
    
    logger.info('final shape in 3mm voxels:{}'.format((np.array(img_vol_cropped.shape)*voxsize_img/target_voxsize_liver_2).astype(int)))
    
    # resize the cropped image
    img_vol_resize = pi.zoom3d(img_vol_cropped, voxsize_img / target_voxsize_liver_1, cval = 0)
    
    # median filter the volume
    img_vol_med = median_filter(img_vol_resize, size = 3) #for U-net
    img_vol_med = pi.zoom3d(img_vol_med, target_voxsize_liver_1 / target_voxsize_liver_2, cval = 0)
    
    # normalize the volume
    if Modality == 'CT': 
      img_vol_med = np.clip(img_vol_med/1000, -0.2, 0.2)*2.5
    elif Modality == 'MR':
      min_vol = np.min(img_vol_med)
      max_vol = np.max(img_vol_med)
      range_vol = max_vol - min_vol
      upper = min_vol+0.8*range_vol
      lower = min_vol
      img_vol_med = np.clip(img_vol_med, lower, upper)
      img_vol_med = (img_vol_med - lower)/(upper - lower) - 0.5
    else:
       logger.error(f'Image modality has to be CT or MR')
       
    # the input image of the CNN model need to be in RAS orientation
    img_vol_med = np.flip(img_vol_med, [0,1])
    
    logger.info('The time for preprocessing {} is {:.2f} s'.format(imgname, time.time() - start_time_tmp))
    
    img_prepro_list.append(img_vol_med)
    
    logger.info('The total time for preprocessing is {:.2f} s'.format(time.time() - start_time_0))
    
    ############################################################################################################################
    #retrieve the trained CNN model and do predictions on the list of numpy arrays which contain the preprocessed CT volumes
    ############################################################################################################################
    
    # use U-net model
    segment_size = [163, 136, 136]
    weight_decay = 1e-5
    center_sampling=True
    cnn_pred_median_list = unet_liver_predict(img_prepro_list,
                                      segment_size,
                                      process_dir,
                                      WholeLiverModel,
                                      output_series_liver_seg_path,
                                      weight_decay    = 1e-5,
                                      center_sampling = center_sampling,
                                      save_output     = False)
               
    ##################################################################################################################
    #post-process the output of the CNN model so that it matches the size of the orignal CT without any preprocessing#
    ##################################################################################################################
    
    start_time_2 = time.time()
    
    cnn_pred_median = np.squeeze(cnn_pred_median_list[0])
    
    logger.info(f'{imgname}')
    
    cnn_liver_pred_pad = cnn_liver_pred_postprocess(cnn_pred_median,
                                        img_vol,
                                        bbox,
                                        Th = 0.5,
                                        flag_flip = True)
    
    structureSetName = f'Liver_seg_{os.path.basename(WholeLiverModel)}__pymirc_{pymirc.__version__}'

    # convert binary mask to RTstruct
    if not input_nifti:
      ofile_liver = os.path.join(output_series_liver_seg_path, imgname + '_cnn_pred_liver.dcm')
      pymf.labelvol_to_rtstruct(cnn_liver_pred_pad, affine_lps, dcm_files, 
                                ofile_liver,
                                seriesDescription = 'liver CNN', 
                                structureSetName  = structureSetName, 
                                roinames          = ['liver CNN'],
                                roidescriptions   = ['liver CNN'],
                                tags_to_add       = {'SeriesDate':date.today()})
      logger.info(f'wrote RTstruct: {ofile_liver}')
    
    #save the binary output as nifti file
    if save_nifti or input_nifti:
      affine_ras       = affine_lps.copy()
      affine_ras[0,-1] = (-1 * affine_lps @ np.array([img_vol.shape[0]-1,0,0,1]))[0]
      affine_ras[1,-1] = (-1 * affine_lps @ np.array([0,img_vol.shape[1]-1,0,1]))[1]
    
      nib.save(nib.Nifti1Image(np.flip(cnn_liver_pred_pad,[0,1]), affine_ras), 
                               os.path.join(output_series_liver_seg_path, imgname + '_cnn_pred_liver.nii'))
    
    logger.info('The total time for postprocessing is {:.2f} s'.format(time.time() - start_time_2))
    logger.info('The total processing time is {:.2f} s'.format(time.time() - start_time_0))

  
  #=========================================================================================================
  #============================================= CNN lesion seg ============================================
  #=========================================================================================================
  if seg_lesion:
    img_prepro_lesion_list = []
    # if liver_seg_dir is None, use CNN liver seg from the previous step
    # if liver_seg_dir is not None, use the liver mask from the doctor
    if liver_seg_dir is None:
      cnn_liv_vol = np.flip(cnn_liver_pred_pad,[0,1])
    else:
      if not input_nifti:
        # read the liver RTst from MIM if the input image is in DICOM format
        rtst_file = glob(os.path.join(liver_seg_dir, '*.dcm'))
        if not rtst_file:
          logger.error(f'There is no liver seg RTst found for {liver_seg_dir}')
        elif len(rtst_file) > 1:
          logger.error(f'There are multiple liver seg RTsts found for {liver_seg_dir}')
        else: 
          img_shape = img_vol.shape
          contour_data = pymf.read_rtstruct_contour_data(rtst_file[0])
          roi_inds = pymf.convert_contour_data_to_roi_indices(contour_data, affine_lps, img_shape, use_contour_orientation = False)
          # create a label array
          cnn_liv_vol = np.zeros(img_shape)              
          for i in range(len(roi_inds)):
            cnn_liv_vol[roi_inds[i]] = int(contour_data[i]['ROINumber'])
          cnn_liv_vol = np.flip(cnn_liv_vol, (0, 1))          
      else:
        liver_seg_files = glob(os.path.join(liver_seg_dir, '*.nii'))
        if len(liver_seg_files) == 1:
          cnn_liv_nii = nib.load(liver_seg_files[0])
          cnn_liv_nii = nib.as_closest_canonical(cnn_liv_nii)
          cnn_liv_vol = cnn_liv_nii.get_data()  
        elif len(liver_seg_files) == 0:
          logger.error(f'No NIFTI file exists in {liver_seg_dir}')
        else:
          logger.error(f'Multiple NIFTI files exist in {liver_seg_dir}')
          
    # normalize the img volume   
    img_vol_ras = np.flip(img_vol, (0, 1))
    if Modality == 'CT':
      img_vol_norm = np.clip(img_vol_ras/1000, -0.2, 0.2)*2.5
    elif Modality == 'MR':      
      min_liv = np.min(img_vol_ras[np.where(cnn_liv_vol > 0)])
      max_liv = np.max(img_vol_ras[np.where(cnn_liv_vol > 0)])
      median_liv = np.median(img_vol_ras[np.where(cnn_liv_vol > 0)])
      #sanity check
      if min_liv > median_liv or max_liv < median_liv:
        logger.error(f'the minimum, median, and maximum values are incorrrect')
    
      img_vol_cent = img_vol_ras - median_liv
      ratio = min(0.5/abs(min_liv-median_liv), 0.5/abs(max_liv-median_liv))
      img_vol_norm = img_vol_cent * ratio
    
    # resize the image and segmentation
    img_vol_resize = pi.zoom3d(img_vol_norm, voxsize_img / target_voxsize_lesion, cval = 0)
    liv_vol_resize = pi.zoom3d(cnn_liv_vol, voxsize_img / target_voxsize_lesion, cval = 0)
    
    liv_vol_resize_bin = liv_vol_resize > 0.5
    liv_vol_resize_bin = liv_vol_resize_bin.astype(dtype='int16')
    
    # find the bbox for liver mask
    struct = ndimage.generate_binary_structure(3,2)
    ss = ndimage.iterate_structure(struct, 2).astype(int)
    liv_vol_dil = ndimage.binary_dilation(liv_vol_resize_bin, structure=ss).astype(int)
    liv_vol_dil = liv_vol_dil.astype('int16')
    # combine liver and tumor in the same mask
    img_mask_liv = img_vol_resize * liv_vol_dil
    img_mask_liv[np.where(liv_vol_dil==0)] = -0.5 
    img_mask_liv_clip = np.clip(img_mask_liv, -0.5, 0.5)
    # crop the images (image is RAS)
    where_x, where_y, where_z = np.where(liv_vol_dil > 0)
    x_start_liver = np.min(where_x)
    x_end_liver = np.max(where_x)
    y_start_liver = np.min(where_y)
    y_end_liver = np.max(where_y)
    z_start_liver = np.min(where_z)
    z_end_liver = np.max(where_z)
    bbox_liver = (slice(x_start_liver, x_end_liver+1, None),
                  slice(y_start_liver, y_end_liver+1, None),
                  slice(z_start_liver, z_end_liver+1, None))
    # crop the masked image
    img_mask_liv_crop = img_mask_liv_clip[bbox_liver]
    # record the size of the resize image before cropping
    img_resize_orig_shape = img_mask_liv_clip.shape
    
    logger.info('The time for preprocessing {} is {:.2f} s'.format(imgname, time.time() - start_time_tmp))
    
    # the input image of the CNN model for lesion seg needs to be in RAS orientation
    img_prepro_lesion_list.append(img_mask_liv_crop)
    
    ############################################################################################################################
    #retrieve the trained CNN model and do predictions on the list of numpy arrays which contain the preprocessed CT volumes
    ############################################################################################################################
    
    # use U-net model
    segment_size = [92,84,42]
    weight_decay = 1e-5
    center_sampling=False
    cnn_lesion_pred_median_list = unet_lesion_predict(img_prepro_lesion_list,
                                      segment_size,
                                      process_dir,
                                      LesionsModel,
                                      output_series_lesion_seg_path,
                                      weight_decay    = 1e-5,
                                      center_sampling = center_sampling,
                                      save_output     = False)
               
    ##################################################################################################################
    #post-process the output of the CNN model so that it matches the size of the orignal CT without any preprocessing#
    ##################################################################################################################
    
    start_time_2 = time.time()
    
    cnn_lesion_pred_median = np.squeeze(cnn_lesion_pred_median_list[0])
    
    logger.info(f'{imgname}')
    
    cnn_lesion_pred_postpro = cnn_lesion_pred_postprocess(cnn_lesion_pred_median,
                                        img_vol,
                                        bbox_liver,
                                        img_resize_orig_shape,
                                        Th = 0.5,
                                        flag_flip = False)
    
    #save the binary output as nifti file
    if save_nifti or input_nifti:
      affine_ras       = affine_lps.copy()
      affine_ras[0,-1] = (-1 * affine_lps @ np.array([img_vol.shape[0]-1,0,0,1]))[0]
      affine_ras[1,-1] = (-1 * affine_lps @ np.array([0,img_vol.shape[1]-1,0,1]))[1]
    
      nib.save(nib.Nifti1Image(cnn_lesion_pred_postpro, affine_ras), 
                               os.path.join(output_series_lesion_seg_path, imgname + '_cnn_pred_liver_lesions.nii'))
    
    if not input_nifti:
      if cnn_lesion_pred_postpro.max() > 0:
        #structureSetName = f'Lesion_seg_{os.path.basename(LesionsModel)}_deepvoxnet_{deepvoxnet.__version__}_pymirc_{pymirc.__version__}'
        structureSetName = f'Lesion_seg_{os.path.basename(LesionsModel)}_pymirc_{pymirc.__version__}'
        # convert padded CNN lesion seg to LPS for conversion to RTst
        cnn_lesion_pred_pad_lps = np.flip(cnn_lesion_pred_postpro, (0, 1))
        # convert binary mask to RTstruct
        ofile = os.path.join(output_series_lesion_seg_path, imgname + '_cnn_pred_liver_lesions.dcm')
        pymf.labelvol_to_rtstruct(cnn_lesion_pred_pad_lps, affine_lps, dcm_files, ofile,
                                  seriesDescription = 'liver lesions CNN', 
                                  structureSetName  = structureSetName, 
                                  roinames          = ['liver lesions CNN'],
                                  roidescriptions   = ['liver lesions CNN'],
                                  tags_to_add       = {'SeriesDate':date.today(),
                                                       'SeriesTime':datetime.now().time()})
      
        logger.info(f'wrote RTstruct: {ofile}')  
      
      else:
        logger.info(f'No liver lesions found. Not creating rtstruct dicom.')  
    
    logger.info('The total time for postprocessing CNN lesion seg is {:.2f} s'.format(time.time() - start_time_2))
    logger.info('The total processing time is {:.2f} s'.format(time.time() - start_time_0))
