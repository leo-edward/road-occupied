# define json data structure
import io
import base64
import json
import numpy as np
from PIL import Image


class DataTransform(object):
    def __init__(self, data):
        self.data = data
        self.taskId = data.get("taskId")
        self.sceneId = data.get("sceneId")
        self.topicName = data.get("topicName")
        self.unionAlgoId = data.get("unionAlgoId")
        self.atomAlgoId = data.get("atomAlgoId")
        self.timestamp = data.get("timestamp")
        self.deviceId = data.get("deviceId")
        self.tenantId = data.get("tenantId")
        self.sourceFile = data.get("sourceFile")
        self.algoParam = data.get("algoParam")
    
    def transform(self, algorithm_result):
        # transform algorithm result to json
        self.result_data = {}
        self.result_data["data"] = {}
        self.origin_image = {}
        self.detection_result = {}
        self.detection_result["attributeMap"] = {}

        self.result_data["unionAlgoId"] = self.unionAlgoId
        self.result_data["atomAlgoId"] = self.atomAlgoId
        self.result_data["sceneId"] = self.sceneId
        self.result_data["timestamp"] = self.timestamp
        self.result_data["deviceId"] = self.deviceId
        self.result_data["tenantId"] = self.tenantId
        
        
        self.origin_image["fileData"]= self.sourceFile
        self.origin_image["attributeMap"]= {}
        self.result_data["data"]["originImage"] = self.origin_image

        self.grapRoiImgs = []
        image_src = base64.b64decode(self.sourceFile)
        image_src = Image.open(io.BytesIO(image_src))
        for i in range(len(algorithm_result.top_label)):
            self.detection_result["box"] = algorithm_result.top_boxes[i].tolist()
            # get the crop image
            image_cropped = image_src.crop(tuple(algorithm_result.top_boxes[i]))
            # encode image_cropped to base64
            image_buffer = io.BytesIO()
            image_cropped.save(image_buffer, format='JPEG')
            base64_str = str(base64.b64encode(image_buffer.getvalue()))
            self.detection_result["fileData"] = base64_str
        
            self.detection_result["score"] = float(algorithm_result.top_conf[i])
            self.detection_result["attributeMap"]["class"] = int(algorithm_result.top_label[i])
            self.detection_result["attributeMap"]["class_name"] = algorithm_result.class_names

            self.grapRoiImgs.append(self.detection_result)
        
        self.result_data["data"]["grapRoiImgs"] = self.grapRoiImgs

        return self.result_data

class AlgorithmResult(object):
    def __init__(self, top_label, top_conf, top_boxes, class_names):
        self.top_label = top_label
        self.top_conf = top_conf
        self.top_boxes = top_boxes
        self.class_names = class_names