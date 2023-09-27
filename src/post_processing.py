import io
import sys
import base64
import colorsys
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont

from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox
from __init__ import *
from src.__init__ import *
import torch
from torchvision.ops import nms

import tritonclient.grpc as grpcclient
from tritonclient import utils


class Image_detect(object):
    def __init__(self):
        self.classes_path = 'model_data/road_occupied_classes.txt'
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        self.input_shape = [640, 640]
        #------------------------------------------------------#
        #   所使用到的yolov7的版本，本仓库一共提供两个：
        #   l : 对应yolov7
        #   x : 对应yolov7_x
        #------------------------------------------------------#
        self.phi = 'l'
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        self.confidence = 0.4
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        self.nms_iou = 0.3
        # triton params
        self.verbose = False
        self.client_timeout = None
        self.grpc_compression_algorithm = None
        self.model_name = "road_occupied"
        self.model_version = ""

    def inference(self, image_src):
        try:
            triton_client = grpcclient.InferenceServerClient(url=triton_url,
                                                            verbose=self.verbose)
        except Exception as e:
            logger.error("Channel creation failed when connect to triton server: " + str(e))
            sys.exit(1)
        self.image_shape = np.array(np.shape(image_src)[0:2])
        image_src = cvtColor(image_src)
        image_data = resize_image(image_src, (640, 640), True)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # Infer
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('input_0', [1, 3, 640, 640], "FP32"))
        inputs[0].set_data_from_numpy(image_data)

        outputs.append(grpcclient.InferRequestedOutput('output_0'))
        outputs.append(grpcclient.InferRequestedOutput('output_1'))
        outputs.append(grpcclient.InferRequestedOutput('output_2'))
         # Test with outputs
        results = triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=self.client_timeout,
            headers={'test': '1'},
            compression_algorithm=self.grpc_compression_algorithm)
        # Get model info
        # statistics = triton_client.get_inference_statistics(model_name=self.model_name)
        # # print(statistics)
        # if len(statistics.model_stats) != 1:
        #     # print("FAILED: Inference Statistics")
        #     logger.error("FAILED: Inference Statistics")
        #     sys.exit(1)
        # Get the output arrays from the results
        output0_data = torch.Tensor(results.as_numpy('output_0'))
        output1_data = torch.Tensor(results.as_numpy('output_1'))
        output2_data = torch.Tensor(results.as_numpy('output_2'))

        results = [output0_data, output1_data, output2_data]
        return results
    
    def result_post_process(self, results):
        class_names, num_classes = get_classes(self.classes_path)
        anchors, num_anchors = get_anchors(self.anchors_path)
        bbox_util = DecodeBox(anchors, num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        outputs = bbox_util.decode_box(results)
        results_decode = bbox_util.non_max_suppression(torch.cat(outputs, 1), num_classes, self.input_shape, 
                        self.image_shape, True, conf_thres = self.confidence, nms_thres = self.nms_iou)
        if results_decode[0] is None:
            return None, None, None, None, None
        top_label = np.array(results_decode[0][:, 6], dtype='int32')
        top_conf = results_decode[0][:, 4] * results_decode[0][:, 5]
        top_boxes = results_decode[0][:, :4]
        return class_names, num_classes, top_label, top_conf, top_boxes

    def draw_image(self, image_src, class_names, num_classes, top_label, top_conf, top_boxes):
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image_src.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image_src.size[0] + image_src.size[1]) // np.mean([640, 640]), 1))

        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image_src.size[1], np.floor(bottom).astype('int32'))
            right   = min(image_src.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image_src)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        image_src.show()
        # encode image_src to base64
        buffered = io.BytesIO()
        image_src.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str

    def image_process(self, message):
        logger.info("start image process...")
        message_data = json.loads(eval(message.data()))
        image_source = message_data['sourceFile']
        # image_source is a base64 string
        image_src = base64.b64decode(image_source)
        image_src = Image.open(io.BytesIO(image_src))
        # Infer
        results = self.inference(image_src)
        class_names, num_classes, top_label, top_conf, top_boxes = self.result_post_process(results)
        if class_names is None:
            return None
        # TODO: add judgement of draw image
        algorithm_result = AlgorithmResult(top_label, top_conf, top_boxes, class_names)
        data_transform = DataTransform(data=message_data)
        result_data = data_transform.transform(algorithm_result)
        logger.info("end image process...")
        return result_data
