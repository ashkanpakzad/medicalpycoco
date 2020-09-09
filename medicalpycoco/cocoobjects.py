import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from pycocotools import mask
from .medicalpycocotools import binary_mask_to_rle, resize_binary_mask

class COCOimage:
    def __init__(self, image_id, image_path,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
        image = Image.open(image_path)
        self.file_name = Path(image_path).name
        [self.width, self.height] = image.size
        self.id = image_id

        self.date_captured = date_captured
        self.license = license_id
        self.coco_url = coco_url
        self.flickr_url = flickr_url

    def todict(self):
        return self.__dict__

class COCOann:
    def __init__(self, annotation_id, image_id, category_info, annotation_filename, image_size=None, bounding_box=None):

        binary_mask = Image.open(annotation_filename)
        if binary_mask.mode != '1' or binary_mask.mode != 'L':
            binary_mask = binary_mask.convert('1')
        binary_mask = np.array(binary_mask).astype(np.uint8)
        # process binary mask
        if image_size is not None:
            binary_mask = resize_binary_mask(binary_mask, image_size)

        binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        area = mask.area(binary_mask_encoded)

        if bounding_box is None:
            bounding_box = mask.toBbox(binary_mask_encoded)

        if category_info["is_crowd"]:
            is_crowd = 1
        else :
            is_crowd = 0

        segmentation = binary_mask_to_rle(binary_mask)

        self.id = annotation_id
        self.image_id= image_id
        self.category_id= category_info["id"]
        self.iscrowd= is_crowd
        self.area= area.tolist()
        self.bbox= bounding_box.tolist()
        self.segmentation= segmentation
        self.width= binary_mask.shape[1]
        self.height= binary_mask.shape[0]

    def todict(self):
        if not self.segmentation:
            # no annotation
            return None
        else:
            return self.__dict__