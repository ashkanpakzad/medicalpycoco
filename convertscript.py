#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from pathlib import Path
from pycocotools import mask
from medicalpycoco.cocoobjects import COCOimage, COCOann

datasetname = 'shapesdataset'  # savename
ROOT_DIR = 'examples/shapes/train'
IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2018")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "description",
    "url": "N/A",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "ashkanpakzad",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'square',
        'supercategory': 'shape',
    },
    {
        'id': 2,
        'name': 'circle',
        'supercategory': 'shape',
    },
    {
        'id': 3,
        'name': 'triangle',
        'supercategory': 'shape',
    },
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image_id = int(Path(image_filename).stem)
            image_info = COCOimage(image_id, image_filename).todict()
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}

                    annotation_info = COCOann(segmentation_id, image_id, category_info, annotation_filename,
                                              image_size=[image_info['width'], image_info['height']]).todict()

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

    with open(f'{ROOT_DIR}/{datasetname}.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
