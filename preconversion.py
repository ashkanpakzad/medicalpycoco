#!/usr/bin/env python3
import os
from medicalpycoco.medicalconverttools import preprocess

# set up paths
ROOT_DIR = 'Task06_Lung/train'
IMAGE_VOL_DIR = os.path.join(ROOT_DIR, "volume")
ANNOTATION_VOL_DIR = os.path.join(ROOT_DIR, "label")

classes = {'1': 'tumour'}
roi = [
    [40, 460], [60, 410], [50, -10]
]

def main():
    img_dir, ann_dir = preprocess(classes, IMAGE_VOL_DIR, ANNOTATION_VOL_DIR, ROOT_DIR, roi=roi, reorient=False)
    print(img_dir)
    print(ann_dir)
if __name__ == "__main__":
    main()
