# The script used to perform CNN liver and/or lesion segmentation

# function definition needed to define entry_points such that script can be installed
def main():
  import pyliverlesionseg
  
  from pathlib import Path
  from argparse import ArgumentParser
  
  # parsing the parameters
  parser = ArgumentParser()
  
  parser.add_argument('input', help = 'the path of input dicom files of the image (CT or MR) or a filename of a nifti file')
  parser.add_argument('--liver_seg_dir', default = None,
                      help = 'directory containing the dicom files of the whole liver RTstruct. If None, this RTstruct will be generated by the WholeLiverModel')
  parser.add_argument('--WholeLiverModel', 
                      default = str(Path(pyliverlesionseg.__file__).parent / 'trained_models' / 'model_unet_ct_mr_liv_seg_resize_1.5mm_med_3_resize_3mm_20201011_dice_loss_val_binary_dice_mean.hdf5'),
                      help = 'the path of the trained CNN model for liver segmentation')
  parser.add_argument('--LesionsModel', 
                      default = str(Path(pyliverlesionseg.__file__).parent / 'trained_models' / 'model_unet_ct_mr_lesion_seg_resize_1mm_1mm_3mm_20220207_dice_loss_val_binary_dice_mean.hdf5'),
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
  
  data_input      = args.input
  liver_seg_dir   = args.liver_seg_dir
  WholeLiverModel = args.WholeLiverModel
  LesionsModel    = args.LesionsModel
  seg_liver       = args.seg_liver
  seg_lesion      = args.seg_lesion
  save_nifti      = args.save_nifti
  input_nifti     = args.input_nifti
  Modality        = args.Modality
  
  # run the main function of liver and lesion segmentation using CNN
  pyliverlesionseg.cnn_liver_lesion_seg_CT_MR_main(data_input, 
                                                   liver_seg_dir, 
                                                   WholeLiverModel, 
                                                   LesionsModel, 
                                                   seg_liver   = seg_liver, 
                                                   seg_lesion  = seg_lesion, 
                                                   save_nifti  = save_nifti, 
                                                   input_nifti = input_nifti, 
                                                   Modality    = Modality)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  main()