from report_utils import *


def report(unfiltred_t1_filename, input_t1_filename, input_flair_filename, MASK_filename, LAB_filename,transform_filename,
            crisp_filename, hemi_fileneame,structures_filename, age='uknown', sex='uknown'):
    FLAIR_img = nii.load(input_flair_filename)
    MASK_img = nii.load(MASK_filename)
    LAB_img = nii.load(LAB_filename)
    LAB =LAB_img.get_fdata()
    #LAB_img = MASK_img
    transform=readITKtransform(transform_filename)
    det = np.linalg.det(transform)
    if (det < 0):
        orientation_report='radiological' # Left is right
    else:
        orientation_report='neurological' # Right is right
    scale = abs(det)
    unfiltred = nii.load(unfiltred_t1_filename).get_fdata()
    filtered = nii.load(input_t1_filename).get_fdata()
    if unfiltred.ndim == 4 and unfiltred.shape[3] == 1:
        unfiltred = np.reshape(unfiltred, unfiltred.shape[:-1])
    snr=compute_SNR(unfiltred, filtered)

    FLAIR=FLAIR_img.get_data()
    LAB=LAB_img.get_data()
    MASK = MASK_img.get_data()
    LAB=LAB.astype('int')
    MASK=MASK.astype('int')
    vol_ice = (compute_volumes(MASK, [[1]], scale))[0]
    CRISP=nii.load(crisp_filename).get_data()
    vols_tissue = (compute_volumes(CRISP, [[1],[2],[3]], scale))

    FLAIR /= 300
    FLAIR = np.clip(FLAIR, 0, 1)
    OUT_HEIGHT=217
    DEFAULT_ALPHA=0.5
    colors_lesions= np.array([[0,0,0], [255,255,0], [0,255,255], [255,0,255], [88,41,0], [249,228,183]])
    colors_tissue= np.array([[0,0,0],[255,0,0], [0,255,0], [0,0,255]])
    colors_ice= np.array([[0,0,0],[255,165,0]])

    lesion_types_filename, region_name, wm_name= get_lesion_by_regions(input_t1_filename, crisp_filename, hemi_fileneame, structures_filename, LAB_filename)
    lesion_types=nii.load(lesion_types_filename).get_data()

    ## Axial
    slice_index=80
    filename_seg_0, filename_ice_0, filename_tissue_0, filename_flair_0 = save_images("0","0",  FLAIR[:, :, slice_index],CRISP[:, :, slice_index],
                                                                                 lesion_types[:, :, slice_index],
                                                                             MASK[:, :, slice_index], colors_ice, colors_lesions,colors_tissue)
    ## Coronal
    slice_index=120
    filename_seg_1, filename_ice_1, filename_tissue_1, filename_flair_1 = save_images("1","1",  FLAIR[:, slice_index, :],CRISP[:, slice_index,:],
                                                                                 lesion_types[:, slice_index, :],
                                                                             MASK[:, slice_index, :], colors_ice, colors_lesions, colors_tissue)
    ## sagittal
    slice_index=70
    filename_seg_2, filename_ice_2, filename_tissue_2, filename_flair_2 = save_images("2","2",  FLAIR[slice_index, :, :],CRISP[ slice_index, :, :],
                                                                                 lesion_types[slice_index, :, :],
                                                                                 MASK[slice_index, :, :], colors_ice, colors_lesions,colors_tissue)


    plot_images_filenames=np.array([[ filename_flair_0,filename_ice_0, filename_tissue_0,filename_seg_0],
     [ filename_flair_1,filename_ice_1, filename_tissue_1,filename_seg_1],
      [ filename_flair_2,filename_ice_2, filename_tissue_2,filename_seg_2 ]])

    filenames_normal_tissue, normal_vol= get_expected_volumes(age, sex, vols_tissue, vol_ice)

    save_pdf(input_t1_filename, age, sex, vols_tissue,vol_ice , snr, orientation_report, filenames_normal_tissue, normal_vol,
    scale, colors_ice, colors_lesions,colors_tissue,lesion_types_filename, plot_images_filenames)

"""
report(unfiltred_t1_filename='to_segment/n_mfmni_ftraining01_01_mprage_pp_check.nii',
input_t1_filename='to_segment/n_mfmni_ftraining01_01_mprage_pp_check.nii',
 input_flair_filename='to_segment/n_mfmni_ftraining01_01_flair_pp_check.nii',
  MASK_filename='to_segment/mask_n_mfmni_ftraining01_01_mprage_pp_check.nii',
   LAB_filename='to_segment/training01_01_mask1.nii',
   transform_filename='to_segment/affine_mftraining01_01_mprage_pp_checkAffine.txt',
    crisp_filename='to_segment/crisp_mfmni_ftraining01_01_mprage_pp_check.nii',
     hemi_fileneame='to_segment/hemi_n_mfmni_ftraining01_01_mprage_pp_check.nii',
     structures_filename='to_segment/lab_n_mfmni_ftraining01_01_mprage_pp_check.nii',
      age='21', sex='female')
"""
