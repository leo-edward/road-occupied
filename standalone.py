import os
import io
import cv2
import time

import numpy as np
from __init__ import *
from PIL import Image

from yolo import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Predict(object):
    def __init__(self):
        self.yolo = YOLO()
    
    def predict(self, image):
        try:
            image = Image.open(io.BytesIO(image))
        except Exception as err:
            logger.error('Open Error! Try again!')
            return err
        else:
            r_image, r_classes, r_scores, r_boxes = self.yolo.detect_image(image)
            return r_image, r_boxes, r_classes, r_scores
            