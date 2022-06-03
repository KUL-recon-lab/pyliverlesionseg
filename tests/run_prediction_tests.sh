python ../scripts/predict_liver_lesion_seg.py ../data/CT-test-01/CT --seg_liver --seg_lesion --save_nifti 
CT_dcm_exit_code=$?
python predict_liver_lesion_seg.py ../data/MR-test-01/MR --seg_liver --seg_lesion --save_nifti 
MR_dcm_exit_code=$?

python ../scripts/predict_liver_lesion_seg.py ../data/CT-test-02/CT.nii --seg_liver --seg_lesion --save_nifti --Modality CT --input_nifti
CT_nii_exit_code=$?
python ../scripts/predict_liver_lesion_seg.py ../data/MR-test-02/MR.nii --seg_liver --seg_lesion --save_nifti --Modality MR --input_nifti
MR_nii_exit_code=$?

echo ""
echo "#-----------------"
echo "CT dcm pred test ${CT_dcm_exit_code}"
echo "MR dcm pred test ${MR_dcm_exit_code}"

echo "CT nii pred test ${CT_nii_exit_code}"
echo "MR nii pred test ${MR_nii_exit_code}"
