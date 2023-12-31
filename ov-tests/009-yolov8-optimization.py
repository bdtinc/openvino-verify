#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is the notebook that was the original source of this test:
# NB_Source: 230-yolov8-object-detection.ipynb

# Put a short description of the test here for the chart display,
# it must be less than 50 characters or it will be truncated:
# Test_Slug_Line: YOLOv8 Object Detection


from ultralytics.yolo.utils import ops
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import Tuple, Dict
import cv2
import numpy as np
from ultralytics.yolo.utils.plotting import colors
from pathlib import Path
import argparse
import sys
import openvino as ov

sys.path.append(str(Path(__file__).resolve().parents[0].joinpath('utils')))
from notebook_utils import download_file

# place the base type of device [CPU, GPU, NPU] as string in UNSUPPORTED_DEVICES list
# if the device is not supported for this script
UNSUPPORTED_DEVICES = []


def main(device):
    data_dir = Path(__file__).resolve().parents[1].joinpath('data')
    models_dir = Path(__file__).resolve().parents[1].joinpath('models')

    def plot_one_box(box:np.ndarray, img:np.ndarray,
                    color:Tuple[int, int, int] = None,
                    label:str = None, line_thickness:int = 5):
        """
        Helper function for drawing single bounding box on image
        Parameters:
            x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
            img (no.ndarray): input image
            color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
            label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
            line_thickness (int, *optional*, 5): thickness for box drawing lines
        """
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        return img


    def draw_results(results:Dict, source_image:np.ndarray, label_map:Dict):
        """
        Helper function for drawing bounding boxes on image
        Parameters:
            image_res (np.ndarray): detection predictions in format [x1, y1, x2, y2, score, label_id]
            source_image (np.ndarray): input image for drawing
            label_map; (Dict[int, str]): label_id to class name mapping
        Returns:
            Image with boxes
        """
        boxes = results["det"]
        for idx, (*xyxy, conf, lbl) in enumerate(boxes):
            label = f'{label_map[int(lbl)]} {conf:.2f}'
            source_image = plot_one_box(xyxy, source_image, label=label, color=colors(int(lbl)), line_thickness=1)
        return source_image

    # Download a test sample
    IMAGE_PATH = data_dir / 'coco_bike.jpg'
    download_file(
        url='https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg',
        filename=IMAGE_PATH.name,
        directory=IMAGE_PATH.parent
    ) 

    DET_MODEL_NAME = "yolov8n"

    det_model = YOLO(models_dir / f'{DET_MODEL_NAME}.pt')
    label_map = det_model.model.names

    res = det_model(IMAGE_PATH)
    Image.fromarray(res[0].plot()[:, :, ::-1])

    # object detection model
    det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
    if not det_model_path.exists():
        det_model.export(format="openvino", dynamic=True, half=False)

    def letterbox(img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
        """
        Resize image and padding for detection. Takes image as input, 
        resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints
        
        Parameters:
        img (np.ndarray): image for preprocessing
        new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
        color (Tuple(int, int, int)): color for filling padded area
        auto (bool): use dynamic input size, only padding for stride constrins applied
        scale_fill (bool): scale image to fill new_shape
        scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
        stride (int): input padding stride
        Returns:
        img (np.ndarray): image after preprocessing
        ratio (Tuple(float, float)): hight and width scaling ratio
        padding_size (Tuple(int, int)): height and width padding size
        
        
        """
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


    def preprocess_image(img0: np.ndarray):
        """
        Preprocess image according to YOLOv8 input requirements. 
        Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.
        
        Parameters:
        img0 (np.ndarray): image for preprocessing
        Returns:
        img (np.ndarray): image after preprocessing
        """
        # resize
        img = letterbox(img0)[0]
        
        # Convert HWC to CHW
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img


    def image_to_tensor(image:np.ndarray):
        """
        Preprocess image according to YOLOv8 input requirements. 
        Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.
        
        Parameters:
        img (np.ndarray): image for preprocessing
        Returns:
        input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range 
        """
        input_tensor = image.astype(np.float32)  # uint8 to fp32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # add batch dimension
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor

    def postprocess(
        pred_boxes:np.ndarray, 
        input_hw:Tuple[int, int], 
        orig_img:np.ndarray, 
        min_conf_threshold:float = 0.25, 
        nms_iou_threshold:float = 0.7, 
        agnosting_nms:bool = False, 
        max_detections:int = 300,
    ):
        """
        YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
        Parameters:
            pred_boxes (np.ndarray): model output prediction boxes
            input_hw (np.ndarray): preprocessed image
            orig_image (np.ndarray): image before preprocessing
            min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
            nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
            agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
            max_detections (int, *optional*, 300):  maximum detections after NMS
        Returns:
        pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label]
        """
        nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
        preds = ops.non_max_suppression(
            torch.from_numpy(pred_boxes),
            min_conf_threshold,
            nms_iou_threshold,
            nc=80,
            **nms_kwargs
        )

        results = []
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append({"det": [], "segment": []})
                continue
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})

        return results

    core = ov.Core()

    det_ov_model = core.read_model(det_model_path)
    if device != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    det_compiled_model = core.compile_model(det_ov_model, device)


    def detect(image:np.ndarray, model:ov.Model):
        """
        OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
        Parameters:
            image (np.ndarray): input image.
            model (Model): OpenVINO compiled model.
        Returns:
            detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
        """
        preprocessed_image = preprocess_image(image)
        input_tensor = image_to_tensor(preprocessed_image)
        result = model(input_tensor)
        boxes = result[model.output(0)]
        input_hw = input_tensor.shape[2:]
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image)
        return detections

    input_image = np.array(Image.open(IMAGE_PATH))
    detections = detect(input_image, det_compiled_model)[0]
    image_with_boxes = draw_results(detections, input_image, label_map)

    Image.fromarray(image_with_boxes)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the MobileNetV3 model using OpenVINO")
    parser.add_argument('--device', default='CPU', help="Specify the device to use (e.g., 'CPU', 'GPU', 'TPU', etc.).")
    args = parser.parse_args()
    DEVICE = args.device

    print(f"Using device: {DEVICE}")
    for item in UNSUPPORTED_DEVICES:
        if item in DEVICE:
            print(f"ERROR: {item} is not supported for this model.")
            # 13 is the return code we are using for a test/device combo that is not supported
            # using exit instead of raising CalledProcessError because the subprocess.check_output
            # call appears to be swallowing the returncode.
            exit(13)
    main(DEVICE)