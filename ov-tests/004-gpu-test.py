#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is the notebook that was the original source of this test:
# NB_Source: 009-gpu-test.ipynb
#
# space seperated list of test groups this test should be added to
# Test_Groups: default
#
# Put a short description of the test here for the chart display,
# it must be less than 50 characters or it will be truncated:
# Test_Slug_Line: GPU test with caching

import argparse
import cv2
import os
import subprocess
import sys
import tarfile
import time
import numpy as np
import openvino as ov
from openvino.tools.mo.front import tf as ov_tf_front
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0].joinpath('utils')))
from notebook_utils import download_file

# place the base type of device [CPU, GPU, NPU] as string in UNSUPPORTED_DEVICES list
# if the device is not supported for this script
UNSUPPORTED_DEVICES = ['CPU', 'NPU', 'GNA']


def main(device, frame_number):
    data_dir = Path(__file__).resolve().parents[1].joinpath('data')
    base_model_dir = Path(__file__).resolve().parents[1].joinpath('models')
    output_dir = Path(__file__).resolve().parents[1].joinpath('output')

    core = ov.Core()
    print(f"AVAILABLE DEVICES: {core.available_devices}")
    print(f"FULL GPU NAME: {core.get_property(device, 'FULL_DEVICE_NAME')}")

    print(f"{device} SUPPORTED_PROPERTIES:\n")
    supported_properties = core.get_property(device, "SUPPORTED_PROPERTIES")
    indent = len(max(supported_properties, key=len))

    for property_key in supported_properties:
        if property_key not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
            try:
                property_val = core.get_property(device, property_key)
            except TypeError:
                property_val = 'UNSUPPORTED TYPE'
            print(f"{property_key:<{indent}}: {property_val}")

    # %%
    # Download a model

    model_name = "ssdlite_mobilenet_v2"
    archive_name = Path(f"{model_name}_coco_2018_05_09.tar.gz")

    # Download the archive
    downloaded_model_path = base_model_dir / archive_name
    if not downloaded_model_path.exists():
        model_url = f"http://download.tensorflow.org/models/object_detection/{archive_name}"
        download_file(model_url, downloaded_model_path.name, downloaded_model_path.parent)

    # Unpack the model
    tf_model_path = base_model_dir / archive_name.with_suffix("").stem / "frozen_inference_graph.pb"
    if not tf_model_path.exists():
        with tarfile.open(downloaded_model_path) as file:
            file.extractall(base_model_dir)

    # Convert the model to OpenVINO IR

    precision = 'FP16'

    # The output path for the conversion.
    model_path = base_model_dir / 'ir_model' / f'{model_name}_{precision.lower()}.xml'

    trans_config_path = Path(ov_tf_front.__file__).parent / "ssd_v2_support.json"
    pipeline_config = base_model_dir / archive_name.with_suffix("").stem / "pipeline.config"

    model = None
    if not model_path.exists():
        model = ov.tools.mo.convert_model(input_model=tf_model_path,
                                        input_shape=[1, 300, 300, 3],
                                        layout='NHWC',
                                        transformations_config=trans_config_path,
                                        tensorflow_object_detection_api_pipeline_config=pipeline_config,
                                        reverse_input_channels=True)
        ov.save_model(model, model_path, compress_to_fp16=(precision == "FP16"))
        print("IR model saved to {}".format(model_path))
    else:
        print("Read IR model from {}".format(model_path))
        model = core.read_model(model_path)

    # %%
    # Skipping various model compiles and caching that were demonstrated already in notebook 102
    # Only doing one pass of the benchmarking external application, this is time consuming and
    # not particularly useful here.

    # External application benchmarking

    bench_device = device

    print(f"Running benchmark_app on {bench_device}...")
    print("This may take a while, please be patient.")
    output = subprocess.check_output(f"benchmark_app -m {model_path} -d {bench_device} \
                                    -hint latency -niter 1000", shell=True)

    print(output.decode('utf-8'))

    # Basic GPU application

    # Read model and compile it on GPU in THROUGHPUT mode
    model = core.read_model(model=model_path)
    device_name = device
    compiled_model = core.compile_model(model=model, device_name=device_name, config={"PERFORMANCE_HINT": "THROUGHPUT"})

    # Get the input and output nodes
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Get the input size
    num, height, width, channels = input_layer.shape
    print('Model input shape:', num, height, width, channels)

    # Load video
    video_file = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4"
    video = cv2.VideoCapture(video_file)
    framebuf = []

    # Go through every frame of video and resize it
    print('Loading video...')
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print('Video loaded!')
            video.release()
            break
        
        # Preprocess frames - convert them to shape expected by model
        input_frame = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
        input_frame = np.expand_dims(input_frame, axis=0)

        # Append frame to framebuffer
        framebuf.append(input_frame)
        

    print('Frame shape: ', framebuf[0].shape)
    print('Number of frames: ', len(framebuf))

    # Define the model's labelmap (this model uses COCO classes)
    classes = [
        "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
        "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush", "hair brush"
    ]

    # Define a callback function that runs every time the asynchronous pipeline completes inference on a frame
    def completion_callback(infer_request: ov.InferRequest, frame_id: int) -> None:
        global frame_number
        stop_time = time.time()
        frame_number += 1

        predictions = next(iter(infer_request.results.values()))
        results[frame_id] = predictions[:10]  # Grab first 10 predictions for this frame
        
        total_time = stop_time - start_time
        frame_fps[frame_id] = frame_number / total_time

    # Create asynchronous inference queue with optimal number of infer requests
    infer_queue = ov.AsyncInferQueue(compiled_model)
    infer_queue.set_callback(completion_callback)

    # Perform inference on every frame in the framebuffer
    results = {}
    frame_fps = {}
    frame_number = 0
    start_time = time.time()
    for i, input_frame in enumerate(framebuf):
        infer_queue.start_async({0: input_frame}, i)

    infer_queue.wait_all()  # Wait until all inference requests in the AsyncInferQueue are completed
    stop_time = time.time()

    # Calculate total inference time and FPS
    total_time = stop_time - start_time
    fps = len(framebuf) / total_time
    time_per_frame = 1 / fps 
    print(f'Total time to infer all frames: {total_time:.3f}s')
    print(f'Time per frame: {time_per_frame:.6f}s ({fps:.3f} FPS)')

    # Set minimum detection threshold
    min_thresh = .6

    # Load video
    video = cv2.VideoCapture(video_file)

    # Get video parameters
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))


    # Create folder and VideoWriter to save output video
    output_dir.mkdir(exist_ok=True)
    # the change here was necessary to get this to work on my ubuntu machine
    # TODO: test this on other platforms before locking it in.
    #fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    #output = cv2.VideoWriter('output/output.mp4', fourcc, fps, (frame_width, frame_height))
    outfile = str(output_dir / 'output.mp4')
    output = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
    # Draw detection results on every frame of video and save as a new video file
    while video.isOpened():
        current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = video.read()
        if not ret:
            print('Video loaded!')
            output.release()
            video.release()
            break
            
        # Draw info at the top left such as current fps, the devices and the performance hint being used
        cv2.putText(frame, f"fps {str(round(frame_fps[current_frame], 2))}", (5, 20), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"device {device_name}", (5, 40), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 1, cv2.LINE_AA) 
        cv2.putText(frame, f"hint {compiled_model.get_property('PERFORMANCE_HINT').name}", (5, 60), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # prediction contains [image_id, label, conf, x_min, y_min, x_max, y_max] according to model
        for prediction in np.squeeze(results[current_frame]):
            if prediction[2] > min_thresh:
                x_min = int(prediction[3] * frame_width)
                y_min = int(prediction[4] * frame_height)
                x_max = int(prediction[5] * frame_width)
                y_max = int(prediction[6] * frame_height)
                label = classes[int(prediction[1])]
                
                # Draw a bounding box with its label above it
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_ITALIC, 1, (255, 0, 0), 1, cv2.LINE_AA)

        output.write(frame)
    print(f"Processessed video saved to: {os.getcwd()}/output/output.mp4")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the MobileNetV3 model using OpenVINO")
    parser.add_argument('--device', default='GPU', help="Specify the device to use (e.g., 'CPU', 'GPU', 'TPU', etc.).")
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

    frame_number = 0
    main(DEVICE, frame_number)