import nibabel as nib
import os
from medicalpycoco.medicalpycocotools import filter_for_img, filter_for_annotations
from PIL import Image
import tifffile
import numpy as np
from scipy import ndimage
from pathlib import Path

def getinstancemasks(binaryimage):
    masks = []
    # get connected components
    cc, _ = ndimage.label(binaryimage)
    num_cc = np.max(cc)
    for i in np.arange(1, num_cc+1): # nonzero values only
        mask = np.zeros_like(binaryimage)
        # mask[np.where(cc == i)] = 1
        mask = cc == i
        masks.append(mask)
    return masks

def preprocess(classes, src_IMAGE_VOL_DIR, src_ANNOTATION_VOL_DIR, newdir, roi=None, reorient=False):
    '''classes must be dict with number keys for corresponding index in images. reorient flag will force images into RAS
    orientation. roi, if set, crops images by index; roi must be a list of 3 elements by 2 specifying the lower and
    upperbounds of each axis. To not specify a crop on a particular axis set to [0][-1].'''
    # TODO: implement roi mechanism to auto crop images, maybe allow mm units and voxels.
# make new train and ann dir
    newdir = Path(newdir)
    imgdir = newdir.joinpath('img')
    anndir = newdir.joinpath('ann')
    newdir.mkdir(parents=True, exist_ok=True)
    imgdir.mkdir(parents=True, exist_ok=True)
    anndir.mkdir(parents=True, exist_ok=True)

# list nifti images
    filetypes = ['*.nii', '*.nii.gz']
    for root, _, files in os.walk(src_IMAGE_VOL_DIR):
        image_vol_files = filter_for_img(root, files, file_types=filetypes)

    # load nifti image
    for image_vol_filename in image_vol_files:
        # identify matching annotation image
        for root, _, files in os.walk(src_ANNOTATION_VOL_DIR):
            ann_vol_file = filter_for_annotations(root, files, image_vol_filename, file_types=filetypes)

        # extract the file name without extension
        imagename = Path(image_vol_filename).name
        for type in filetypes:
            if imagename.endswith(type[1:]):
                imagename = imagename.rstrip(type[1:])
                break

        # load nifti images
        img = nib.load(image_vol_filename)
        ann = nib.load(ann_vol_file[0])
        if reorient:
            # reorient to RAS
            img = nib.as_closest_canonical(img)
            ann = nib.as_closest_canonical(ann)

        # convert image slice by slice to TIFF
        vol = img.get_fdata()
        if roi:
            vol = vol[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1], roi[2][0]:roi[2][1]]
        for i in range(vol.shape[-1]):
            slice = vol[..., i]
            if roi:
                slice = slice[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
            filesavename = str(imagename)+'_'+str(i)+'.tiff'
            tifffile.imwrite(imgdir.joinpath(filesavename), slice.astype(np.int16))
            print(filesavename)

        # convert slice by slice by instance by class to TIFF
        annvol = ann.get_fdata()
        if roi:
            annvol = annvol[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1], roi[2][0]:roi[2][1]]
        for i in range(annvol.shape[-1]):
            slice = annvol[..., i].astype(np.int)
            if roi:
                slice = slice[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
            # per class
            for j in range(1, np.max(slice)+1):
                classslice = slice == j
                # per instance
                masks = getinstancemasks(classslice)
                if masks:
                    instanceidx = 0
                    for mask in masks:
                        classname = classes[str(j)]
                        filesavename = str(imagename)+'_'+str(i)+'_'+classname+'_'+str(instanceidx)+'.png'
                        ann_pil = Image.fromarray(mask.astype(np.uint8), mode='L')
                        ann_pil.save(anndir.joinpath(filesavename))
                        print(filesavename)
                        instanceidx =+ 1
    return imgdir, anndir
