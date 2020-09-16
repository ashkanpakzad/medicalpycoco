#!/usr/bin/env python3
import os
import re
import numpy as np
from PIL import Image
import fnmatch
from pathlib import Path
from .cocoobjects import COCOimage, COCOann

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]


# filter images methods

def filter_for_img(root, files, file_types=None):
    if file_types is None:
        file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files


def filter_for_annotations(root, files, image_filename, file_types=None):
    if file_types is None:
        file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files

def convert2coco(coco_header, IMAGE_DIR, ANNOTATION_DIR, img_file_types=None, ann_file_types=None ):
    coco_output = coco_header.copy()
    CATEGORIES = coco_header["categories"]
    segmentation_id = 1
    image_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_img(root, files, file_types=img_file_types)

        # go through each image
        for image_filename in image_files:
            image_info = COCOimage(image_id, image_filename).todict()
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename, file_types=ann_file_types)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if '_'+x['name']+'_' in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}

                    annotation_info = COCOann(segmentation_id, image_id, category_info, annotation_filename,
                                              image_size=[image_info['width'], image_info['height']]).todict()

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1
            image_id = image_id + 1

    return coco_output
