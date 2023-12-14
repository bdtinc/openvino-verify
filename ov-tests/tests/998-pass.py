#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is the notebook that was the original source of this test:
# NB_Source: 001-Hello-World.ipynb

# Put a short description of the test here for the chart display,
# it must be less than 50 characters or it will be truncated:
# Test_Slug_Line: Empty Test

from pathlib import Path
import argparse
import cv2
import sys
import numpy as np
import openvino as ov

# place the base type of device [CPU, GPU, NPU] as string in UNSUPPORTED_DEVICES list
# if the device is not supported for this script
UNSUPPORTED_DEVICES = []


def main(device):
    pass

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