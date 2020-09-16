from medicalpycoco.medicalconverttools import datadivider
import os

ROOT_DIR = 'Task08_HepaticVessel'
IMAGE_DIR = os.path.join(ROOT_DIR, "imagesTr")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "labelsTr")
proportions = [0.7, 0.1, 0.2] # must add up to 1

datadivider(IMAGE_DIR, ANNOTATION_DIR, ROOT_DIR, proportions, seed=None)