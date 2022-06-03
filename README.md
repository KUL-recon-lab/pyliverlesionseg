## pyliverlesionseg - CNN-based liver and liver lesion segmentation in CT and MR

* pyliverlesionseg is a Python framework based on Keras and Tensorflow to perform voxel-based liver and lesion segmentation. It is built based on [DeepVoxNet version 1](https://github.com/JeroenBertels/deepvoxnet), which is a deep learning processing framework for Keras and developed in medical imaging research center (MIRC) of KU Leuven.<br/>
* It allows training on segments (i.e. patches, subvolumes) of an image. Segments are identified by the subject_id and the coordinate of the center voxel.

### References
1. Robben, D., Bertels, J., Willems, S., Vandermeulen, D., Maes, F., Suetens, P. (2018). *DeepVoxNet: voxel‐wise prediction for 3D images.* Report No. KUL/ESAT/PSI/1801.
2. Tang, X. et al. *Whole liver segmentation based on deep learning and manual adjustment for clinical use in SIRT*, European journal of nuclear medicine and molecular imaging, 47, 2020, [link](https://link.springer.com/article/10.1007/s00259-020-04800-3) 


---

### Installation

#### Installation from conda-forge or pypi

* TODO (as soon as package is on pypi / conda forge)

#### Dowloading from github 

An alternative to install the package (especially for developers) is to clone this repository or to download on of github releases. When using this method make sure that environment variable ```PYTHONPATH``` contains the location where the package was cloned / downloaded.

---

### Model prediction for liver segmentation
* The input image is in DICOM format<br/>
python predict_liver_lesion_seg.py <data_path> --seg_liver

* The input image (e.g. CT) is in NIFTI format<br/>
python predict_liver_lesion_seg.py <data_path> --seg_liver --input_nifti --Modality CT


### Model prediction for lesion segmentation
* The input image is in DICOM format<br/>
python predict_liver_lesion_seg.py <data_path> --seg_lesion

* The input image (e.g. CT) is in NIFTI format<br/>
python predict_liver_lesion_seg.py <data_path> --seg_lesion --input_nifti --Modality CT

---

### Model training for liver segmentation
* The default parameters of the script 'train_liver_lesion_seg.py' are for CNN liver segmentation training. For CNN lesion segmentation training, new values of some parameters need to be given.

* The input image for the CNN model is in NIFTI format.

* pre-processing: the raw image needs to be pre-processed before being put into the CNN model. The codes for pre-processing the raw image can be found in pyliverlesionseg/CNN_liver_lesion_seg_CT_MR_functions.py.
  1. The raw image should first be cropped so that the cropped image only contains the whole abdomen in the transaxial slice and the full liver in the z direction. This via the function 'crop_ct_image' (for CT) or 'crop_mr_image' (for MR) in pyliverlesionseg.CNN_liver_lesion_seg_CT_MR_functions.py. These two functions will generate a bounding box saved in an excel file. You can also define a bounding box by yourself.
  2. After that, the cropped image needs to be resampled to an isotropic voxel size of 3 mm. 
  3. The cropped and resampled CT needs to be clipped between -200 HU and 200 HU and normalized through linear mapping to an intensity range of [-0.5, 0.5]. The cropped and resampled MR needs to be clipped between the minimal intensity of the MR and minimum intensity + 0.8 * the intensity range of the MR and normalized through linear mapping to an intensity range of [-0.5, 0.5].
  4. The ground truth liver segmentation should also be cropped by using the bounding box and and resampled to an isotropic voxel size of 3 mm.

* The script for training a CNN for liver segmentation can be run via:<br/>
python train_liver_lesion_seg.py

### Model training for liver lesion segmentation
* The input image for the CNN model is in NIFTI format.

* pre-processing: the raw image needs to be pre-processed before being put into the CNN model. The codes for pre-processing the raw image can be found in pyliverlesionseg/CNN_liver_lesion_seg_CT_MR_functions.py.
  1. The raw CT needs to be clipped between -200 HU and 200 HU and normalized through linear mapping to an intensity range of [-0.5, 0.5]. The raw MR intensities are subtracted by the median intensity of the MR inside the liver mask, clipped between the minimum and maximum intensity of the centralized MR inside the liver mask, and normalized through linear mapping to an intensity range of [-0.5, 0.5].
  2. The normalized image is resampled to a voxel size of [1 mm, 1 mm, 3 mm].
  3. The resampled image is masked by a resampled liver mask and cropped by using the bounding box of the resampled liver mask. The image intensities outside the resampled liver mask is set to -0.5.
  4. The ground truth lesion segmentation should also be resampled to a voxel size of [1 mm, 1 mm, 3 mm] and cropped by using the bounding box of the resampled liver mask.

* The script for training a CNN for lesion segmentation can be run via:<br/>
python train_liver_lesion_seg.py --data_path <data_path> --nb_subjects 180 --training_indice_range 145 --validation_indice_range 145 180 --run_folder_name Runs_lesion_seg_output_size_92_84_42 --network_architecture_id 2 --segment_size 92 84 42 --no_center_sampling --sgd_batch_size 4 --prediction_batch_size 4 --nb_samples_training 320 --nb_samples_validation 140 --max_number_of_subjects_used_for_training 80 --max_number_of_subjects_used_for_validation 35 --nb_subepochs 5 

---

### Acknowledgements
This project has received funding from the European Horizon2020 ITN project (HYBRID, MSCA 764458, https://www.hybrid2020.eu/home.html) and the Research Foundation Flanders (FWO, grant G082418N).
