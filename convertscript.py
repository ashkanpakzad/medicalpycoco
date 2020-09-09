#!/usr/bin/env python3
import datetime
import json
import os
from medicalpycoco.medicalpycocotools import convert2coco

# set up paths
datasetname = 'shapesdataset'  # savename
ROOT_DIR = 'examples/shapes/train'
IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2018")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

# Data types
img_file_types = None  # Default is ['*.jpg', '*.jpeg']
ann_file_types = None  # Default is ['*.png']

# set up COCO header
INFO = {
    "description": "description",
    "url": "N/A",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "collective",
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


def main():
    coco_header = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    coco_output = convert2coco(coco_header, IMAGE_DIR, ANNOTATION_DIR,
                               img_file_types=img_file_types, ann_file_types=ann_file_types)

    with open(f'{ROOT_DIR}/{datasetname}.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == "__main__":
    main()
