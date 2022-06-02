#--------------------------------------------------------------------------------------------------------------
# The script used to perform CNN liver and/or lesion segmentation
#--------------------------------------------------------------------------------------------------------------

import os
import time
from argparse import ArgumentParser

from pyliverlesionseg.CNN_liver_lesion_seg_CT_MR_functions import CNN_liver_lesion_seg_CT_MR_main

# parsing the parameters
parser = ArgumentParser()

parser.add_argument('--process_dir', 
                    default = '../data/1.2.124._11074481_MR_2021-09-16_111317_RAD.ex.mr.abdomen,(1-2-5-17)_t1.starvibe.dixon.tra.in.out.opp_n72__00000/image_dicom', 
                    help = 'the path of input dicom files of the image (CT or MR)')
parser.add_argument('--liver_seg_dir', 
                    default = None,
                    help = 'directory containing the dicom files of the whole liver RTstruct. If None, this RTstruct will be generated by the WholeLiverModel')
parser.add_argument('--WholeLiverModel', 
                    default ='../data/model_unet_ct_mr_liv_seg_resize_1.5mm_med_3_resize_3mm_20201011_dice_loss_val_binary_dice_mean.hdf5',
                    help = 'the path of the trained CNN model for liver segmentation')
parser.add_argument('--LesionsModel', 
                    default = '../data/model_unet_ct_mr_lesion_seg_resize_1mm_1mm_3mm_20220207_dice_loss_val_binary_dice_mean.hdf5',
                    help = 'the path of the trained CNN model for lesion segmentation')
parser.add_argument('--seg_liver', 
                    help = 'whether to perform CNN liver segmentation', action = 'store_true')      
parser.add_argument('--seg_lesion', 
                    help = 'whether to perform CNN lesion segmentation', action = 'store_true')                           
parser.add_argument('--save_nifti', 
                    help = 'whether to save the CNN segmentation as nifti file', action = 'store_true')
parser.add_argument('--input_nifti', 
                    help = 'whether to use the input image in DICOM or NIFTI format. If true, the input image is in NIFTI format', action = 'store_true')
parser.add_argument('--Modality', 
                    default = None,
                    help = 'CT or MR, default = None. When input_nifti is true, Modality has to be given')
                    
args = parser.parse_args()

process_dir = args.process_dir
liver_seg_dir  = args.liver_seg_dir
WholeLiverModel = args.WholeLiverModel
LesionsModel = args.LesionsModel
seg_liver = args.seg_liver
seg_lesion = args.seg_lesion
save_nifti = args.save_nifti
input_nifti = args.input_nifti
Modality = args.Modality

# run the main function of liver and lesion segmentation using CNN
CNN_liver_lesion_seg_CT_MR_main(process_dir, liver_seg_dir, WholeLiverModel, LesionsModel, seg_liver = seg_liver, seg_lesion = seg_lesion, save_nifti = save_nifti, input_nifti = input_nifti, Modality = Modality)