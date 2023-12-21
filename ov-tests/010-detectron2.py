#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is the notebook that was the original source of this test:
# NB_Source: 123-detectron2-to-openvino.ipynb

# Put a short description of the test here for the chart display,
# it must be less than 50 characters or it will be truncated:
# Test_Slug_Line: Detectron2 Conversion

from detectron2.structures import Instances, Boxes
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import detectron2.data.transforms as T
from detectron2.data import detection_utils
import torch
import requests
from PIL import Image
from detectron2.modeling import GeneralizedRCNN
from detectron2.export import TracingAdapter
import torch
import openvino as ov
import warnings
from typing import List, Dict
import detectron2.model_zoo as detectron_zoo
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
    DATA_DIR = Path(__file__).resolve().parents[1].joinpath('data')
    MODEL_DIR = Path(__file__).resolve().parents[1].joinpath('models')


    def get_model_and_config(model_name:str):
        """
        Helper function for downloading PyTorch model and its configuration from Detectron2 Model Zoo

        Parameters:
        model_name (str): model_id from Detectron2 Model Zoo
        Returns:
        model (torch.nn.Module): Pretrained model instance
        cfg (Config): Configuration for model
        """
        cfg = detectron_zoo.get_config(model_name + '.yaml', trained=True)
        model = detectron_zoo.get(model_name + '.yaml', trained=True)
        return model, cfg

    def convert_detectron2_model(model:torch.nn.Module, sample_input:List[Dict[str, torch.Tensor]]):
        """
        Function for converting Detectron2 models, creates TracingAdapter for making model tracing-friendly,
        prepares inputs and converts model to OpenVINO Model

        Parameters:
        model (torch.nn.Module): Model object for conversion
        sample_input (List[Dict[str, torch.Tensor]]): sample input for tracing
        Returns:
        ov_model (ov.Model): OpenVINO Model
        """
        # prepare input for tracing adapter
        tracing_input = [{'image': sample_input[0]["image"]}]

        # override model forward and disable postprocessing if required
        if isinstance(model, GeneralizedRCNN):
            def inference(model, inputs):
                # use do_postprocess=False so it returns ROI mask
                inst = model.inference(inputs, do_postprocess=False)[0]
                return [{"instances": inst}]
        else:
            inference = None  # assume that we just call the model directly

        # create traceable model
        traceable_model = TracingAdapter(model, tracing_input, inference)
        warnings.filterwarnings("ignore")
        # convert PyTorch model to OpenVINO model
        ov_model = ov.convert_model(traceable_model, example_input=sample_input[0]["image"])
        return ov_model

    input_image_url = "https://farm9.staticflickr.com/8040/8017130856_1b46b5f5fc_z.jpg"

    image_file = DATA_DIR / "example_image.jpg"

    if not image_file.exists():
        image = Image.open(requests.get(input_image_url, stream=True).raw)
        image.save(image_file)
    else:
        image = Image.open(image_file)

    def get_sample_inputs(image_path, cfg):
        # get a sample data
        original_image = detection_utils.read_image(image_path, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs

    model_name = 'COCO-Detection/faster_rcnn_R_50_FPN_1x'
    model, cfg = get_model_and_config(model_name)
    sample_input = get_sample_inputs(image_file, cfg)

    model_xml_path = MODEL_DIR / (model_name.split("/")[-1] + '.xml')
    if not model_xml_path.exists():
        ov_model = convert_detectron2_model(model, sample_input)
        ov.save_model(ov_model, MODEL_DIR / (model_name.split("/")[-1] + '.xml'))
    else:
        ov_model = model_xml_path

    core = ov.Core()

    compiled_model = core.compile_model(ov_model, device)
    results = compiled_model(sample_input[0]["image"])

    def postprocess_detection_result(outputs:Dict, orig_height:int, orig_width:int, conf_threshold:float = 0.0):
        """
        Helper function for postprocessing prediction results

        Parameters:
        outputs (Dict): OpenVINO model output dictionary
        orig_height (int): original image height before preprocessing
        orig_width (int): original image width before preprocessing
        conf_threshold (float, optional, defaults 0.0): confidence threshold for valid prediction
        Returns:
        prediction_result (instances): postprocessed predicted instances
        """
        boxes = outputs[0]
        classes = outputs[1]
        has_mask = len(outputs) >= 5
        masks = None if not has_mask else outputs[2]
        scores = outputs[2 if not has_mask else 3]
        model_input_size = (int(outputs[3 if not has_mask else 4][0]), int(outputs[3 if not has_mask else 4][1]))
        filtered_detections = scores >= conf_threshold
        boxes = Boxes(boxes[filtered_detections])
        scores = scores[filtered_detections]
        classes = classes[filtered_detections]
        out_dict = {"pred_boxes": boxes, "scores": scores, "pred_classes": classes}
        if masks is not None:
            masks = masks[filtered_detections]
            out_dict["pred_masks"] = torch.from_numpy(masks)
        instances = Instances(model_input_size, **out_dict)
        return detector_postprocess(instances, orig_height, orig_width)

    def draw_instance_prediction(img:np.ndarray, results:Instances, cfg:"Config"):
        """
        Helper function for visualization prediction results

        Parameters:
        img (np.ndarray): original image for drawing predictions
        results (instances): model predictions
        cfg (Config): model configuration
        Returns:
        img_with_res: image with results   
        """
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        visualizer = Visualizer(img, metadata, instance_mode=ColorMode.IMAGE)
        img_with_res = visualizer.draw_instance_predictions(results)
        return img_with_res

    results = postprocess_detection_result(results, sample_input[0]["height"], sample_input[0]["width"], conf_threshold=0.05)
    img_with_res = draw_instance_prediction(np.array(image), results, cfg)
    Image.fromarray(img_with_res.get_image())


    model_name = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x"
    model, cfg = get_model_and_config(model_name)
    sample_input = get_sample_inputs(image_file, cfg)

    model_xml_path = MODEL_DIR / (model_name.split("/")[-1] + '.xml')

    if not model_xml_path.exists():
        ov_model = convert_detectron2_model(model, sample_input)
        ov.save_model(ov_model, MODEL_DIR / (model_name.split("/")[-1] + '.xml'))
    else:
        ov_model = model_xml_path


    compiled_model = core.compile_model(ov_model, device)
    results = compiled_model(sample_input[0]["image"])
    results = postprocess_detection_result(results, sample_input[0]["height"], sample_input[0]["width"], conf_threshold=0.05)
    img_with_res = draw_instance_prediction(np.array(image), results, cfg)
    Image.fromarray(img_with_res.get_image())


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