#!/usr/bin/env python3
import os
from medicalpycoco.medicalconverttools import preprocess

# set up paths
ROOT_DIR = 'Task08_HepaticVessel/test'
IMAGE_VOL_DIR = os.path.join(ROOT_DIR, "volume")
ANNOTATION_VOL_DIR = os.path.join(ROOT_DIR, "label")

classes = {'1': 'vessel', '2': 'tumour'}
roi = [
    [100, 420], [60, 440], [2, -2]
]
merge = True # merge multiple instances in a single slice into one.

def main():
    img_dir, ann_dir = preprocess(classes, IMAGE_VOL_DIR, ANNOTATION_VOL_DIR, ROOT_DIR, roi=roi, merge=merge, reorient=True)
    print(img_dir)
    print(ann_dir)
if __name__ == "__main__":
    main()
