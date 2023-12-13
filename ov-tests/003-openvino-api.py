#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# 
# This is the notebook that was the original source of this test:
# NB_Source: 002-openvino-api.ipynb

# Put a short description of the test here for the chart display,
# it must be less than 50 characters or it will be truncated:
# Test_Slug_Line: OpenVINO API, format transfoms


import argparse
import cv2
import numpy as np
import openvino as ov
from pathlib import Path
import sys
import time
import urllib.request

sys.path.append(str(Path(__file__).resolve().parents[0].joinpath('utils')))
from notebook_utils import download_file

# place the base type of device [CPU, GPU, NPU] as string in UNSUPPORTED_DEVICES list
# if the device is not supported for this script
UNSUPPORTED_DEVICES = []

def print_available_devices():
    core = ov.Core()

    devices = core.available_devices

    for device in devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")


def main(device):
    core = ov.Core()
    # check devices
    print_available_devices()
    
    base_model_dir = Path(__file__).resolve().parents[1].joinpath('models')
    data_dir = Path(__file__).resolve().parents[1].joinpath('data')

    # OpenVINO IR Model
    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'classification.xml'
    ir_model_name_bin = 'classification.bin'

    download_file(ir_model_url + ir_model_name_xml, filename='ovir_' + ir_model_name_xml, directory=base_model_dir)
    download_file(ir_model_url + ir_model_name_bin, filename='ovir_' + ir_model_name_bin, directory=base_model_dir)

    classification_model_xml = base_model_dir / "ovir_classification.xml"

    model = core.read_model(model=classification_model_xml)
    compiled_model = core.compile_model(model=model, device_name=device)

    # ONYX Model

    onnx_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/segmentation.onnx'
    onnx_model_name = 'segmentation.onnx'

    download_file(onnx_model_url, filename=onnx_model_name, directory=base_model_dir)

    onnx_model_path = base_model_dir / "segmentation.onnx"

    model_onnx = core.read_model(model=onnx_model_path)
    compiled_model_onnx = core.compile_model(model=model_onnx, device_name=device)
    ov.save_model(model_onnx, output_model= base_model_dir / "exported_onnx_model.xml")

    # PadddlePaddle Model

    paddle_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    paddle_model_name = 'inference.pdmodel'
    paddle_params_name = 'inference.pdiparams'

    download_file(paddle_model_url + paddle_model_name, filename=paddle_model_name, directory=base_model_dir)
    download_file(paddle_model_url + paddle_params_name, filename=paddle_params_name, directory=base_model_dir)

    paddle_model_path = base_model_dir / "inference.pdmodel"

    model_paddle = core.read_model(model=paddle_model_path)
    compiled_model_paddle = core.compile_model(model=model_paddle, device_name=device)
    ov.save_model(model_paddle, output_model=base_model_dir / "exported_paddle_model.xml")

    # TensorFlow Model

    pb_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/classification.pb'
    pb_model_name = 'classification.pb'

    download_file(pb_model_url, filename=pb_model_name, directory=base_model_dir)

    tf_model_path = base_model_dir / "classification.pb"

    model_tf = core.read_model(model=tf_model_path)
    compiled_model_tf = core.compile_model(model=model_tf, device_name=device)
    ov.save_model(model_tf, output_model=base_model_dir / "exported_tf_model.xml")

    # TensorFlow Lite Model

    tflite_model_url = 'https://www.kaggle.com/models/tensorflow/inception/frameworks/tfLite/variations/v4-quant/versions/1?lite-format=tflite'
    tflite_model_path = base_model_dir / 'classification.tflite'

    download_file(tflite_model_url, filename=tflite_model_path.name, directory=tflite_model_path.parent)

    model_tflite = core.read_model(tflite_model_path)
    compiled_model_tflite = core.compile_model(model=model_tflite, device_name=device)
    ov.save_model(model_tflite, output_model=base_model_dir / "exported_tflite_model.xml")

    # Getting Model Information

    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'classification.xml'
    ir_model_name_bin = 'classification.bin'

    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=base_model_dir)
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=base_model_dir)

    # Inputs

    classification_model_xml = base_model_dir / "classification.xml"
    model = core.read_model(model=classification_model_xml)
    print('Model Inputs: ', model.inputs)

    input_layer = model.input(0)
    print(f'Input Layer Name: {input_layer.any_name}')
    print(f"input precision: {input_layer.element_type}")
    print(f"input shape: {input_layer.shape}")



    classification_model_xml = base_model_dir / "classification.xml"
    model = core.read_model(model=classification_model_xml)
    print(f"Model Outputs: {model.outputs}")

    output_layer = model.output(0)
    print(f"Output Layer Name: {output_layer.any_name}")
    print(f"output precision: {output_layer.element_type}")
    print(f"output shape: {output_layer.shape}")
    
    # Inference On Model

    # load network
    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'classification.xml'
    ir_model_name_bin = 'classification.bin'

    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=base_model_dir)
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=base_model_dir)

    classification_model_xml = base_model_dir / "classification.xml"
    model = core.read_model(model=classification_model_xml)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # load image and convert to input shape
    image_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_hollywood.jpg",
        directory=data_dir
    )
    image = cv2.imread(str(image_filename))
    print(f"Image Shape: {image.shape}")

    # N,C,H,W = batch size, number of channels, height, width.
    N, C, H, W = input_layer.shape
    # OpenCV resize expects the destination size as (width, height).
    resized_image = cv2.resize(src=image, dsize=(W, H))
    print(f"Resized Image Shape: {resized_image.shape}")

    input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
    print(f"NCHW Formatted Image Shape: {input_data.shape}")

    # execute inference

    # for single input models only
    result = compiled_model(input_data)[output_layer]

    # for multiple inputs in a list
    result = compiled_model([input_data])[output_layer]

    # or using a dictionary, where the key is input tensor name or index
    result = compiled_model({input_layer.any_name: input_data})[output_layer]

    request = compiled_model.create_infer_request()
    request.infer(inputs={input_layer.any_name: input_data})
    result = request.get_output_tensor(output_layer.index).data
    print(f"Inference Result Shape: {result.shape}")

    # Reshaping and Resizing

    # Change Image Size
    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'segmentation.xml'
    ir_model_name_bin = 'segmentation.bin'

    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=base_model_dir)
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=base_model_dir)

    segmentation_model_xml = base_model_dir / "segmentation.xml"
    segmentation_model = core.read_model(model=segmentation_model_xml)
    segmentation_input_layer = segmentation_model.input(0)
    segmentation_output_layer = segmentation_model.output(0)

    print("~~~~ ORIGINAL MODEL ~~~~")
    print(f"input shape: {segmentation_input_layer.shape}")
    print(f"output shape: {segmentation_output_layer.shape}")

    new_shape = ov.PartialShape([1, 3, 544, 544])
    segmentation_model.reshape({segmentation_input_layer.any_name: new_shape})
    segmentation_compiled_model = core.compile_model(model=segmentation_model, device_name=device)
    
    print("~~~~ RESHAPED MODEL ~~~~")
    print(f"model input shape: {segmentation_input_layer.shape}")
    print(
        f"compiled_model input shape: "
        f"{segmentation_compiled_model.input(index=0).shape}"
    )
    print(f"compiled_model output shape: {segmentation_output_layer.shape}")

    # change batch size
    segmentation_model_xml = base_model_dir / "segmentation.xml"
    segmentation_model = core.read_model(model=segmentation_model_xml)
    segmentation_input_layer = segmentation_model.input(0)
    segmentation_output_layer = segmentation_model.output(0)
    new_shape = ov.PartialShape([2, 3, 544, 544])
    segmentation_model.reshape({segmentation_input_layer.any_name: new_shape})
    segmentation_compiled_model = core.compile_model(model=segmentation_model, device_name=device)
    print("~~~~ REBATCHED MODEL ~~~~")
    print(f"input shape: {segmentation_input_layer.shape}")
    print(f"output shape: {segmentation_output_layer.shape}")

    # propagate through network
    segmentation_model_xml = base_model_dir / "segmentation.xml"
    segmentation_model = core.read_model(model=segmentation_model_xml)
    segmentation_input_layer = segmentation_model.input(0)
    segmentation_output_layer = segmentation_model.output(0)
    new_shape = ov.PartialShape([2, 3, 544, 544])
    segmentation_model.reshape({segmentation_input_layer.any_name: new_shape})
    segmentation_compiled_model = core.compile_model(model=segmentation_model, device_name=device)
    input_data = np.random.rand(2, 3, 544, 544)

    output = segmentation_compiled_model([input_data])

    print("~~~~ PROPAGATED MODEL ~~~~")
    print(f"input data shape: {input_data.shape}")
    print(f"result data data shape: {segmentation_output_layer.shape}")
    # %%
    # model caching
    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'classification.xml'
    ir_model_name_bin = 'classification.bin'

    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=base_model_dir)
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=base_model_dir)

    # removed gpu caching test because it is in the notebook 108-gpu-device.ipynb

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
