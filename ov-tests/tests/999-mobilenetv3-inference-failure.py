#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# This is the notebook that was the original source of this test:
# NB_Source: 001-Hello-World.ipynb

# Put a short description of the test here for the chart display,
# it must be less than 50 characters or it will be truncated:
# Test_Slug_Line: MobileNetV3 Inference Failure

from pathlib import Path
import argparse
import cv2
import numpy as np
import openvino as ov
from notebook_utils import download_file

# place the base type of device [CPU, GPU, NPU] as string in UNSUPPORTED_DEVICES list
# if the device is not supported for this script
UNSUPPORTED_DEVICES = []
def main(device):
    base_artifacts_dir = Path('./artifacts')

    model_name = "v3-small_224_1.0_float"
    model_xml_name = f'{model_name}.xml'
    model_bin_name = f'{model_name}.bin'

    model_xml_path = base_artifacts_dir / model_xml_name

    base_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/'

    if not model_xml_path.exists():
        download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
        download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
    else:
        print(f'{model_name} already downloaded to {base_artifacts_dir}')

    
    core = ov.Core()
    model = core.read_model(model=model_xml_path)
    # putting quotes around the device name is necessary to make this fail for testing.
    compiled_model = core.compile_model(model=model, device_name='device')

    output_layer = compiled_model.output(0)

    # Download the image from the openvino_notebooks storage
    image_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        directory="data"
    )

    # The MobileNet model expects images in RGB format.
    image = cv2.cvtColor(cv2.imread(filename=str(image_filename)), code=cv2.COLOR_BGR2RGB)

    # Resize to MobileNet image shape.
    input_image = cv2.resize(src=image, dsize=(224, 224))

    # Reshape to model input shape.
    input_image = np.expand_dims(input_image, 0)

    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)

    imagenet_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
        directory="data"
    )

    imagenet_classes = imagenet_filename.read_text().splitlines()
    # The model description states that for this model, class 0 is a background.
    # Therefore, a background must be added at the beginning of imagenet_classes.
    imagenet_classes = ['background'] + imagenet_classes

    print(imagenet_classes[result_index])

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