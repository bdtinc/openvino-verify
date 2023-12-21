#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is the notebook that was the original source of this test:
# NB_Source: 124-hugging-face-hub.ipynb
#
# space seperated list of test groups this test should be added to
# Test_Groups: default
#
# Put a short description of the test here for the chart display,
# it must be less than 50 characters or it will be truncated:
# Test_Slug_Line: Hugging Face Transformer with Optimum Intel

# External Requirements:
# %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu "transformers[torch]>=4.33.0"
# %pip install -q "optimum-intel"@git+https://github.com/huggingface/optimum-intel.git onnx
#

import argparse
from pathlib import Path
import numpy as np
import subprocess
import openvino as ov
from optimum.intel.openvino import OVModelForSequenceClassification
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# %%
UNSUPPORTED_DEVICES = []

def main(device):
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    base_model_dir = Path(__file__).resolve().parents[1].joinpath('models')

    tokenizer = AutoTokenizer.from_pretrained(MODEL, return_dict=True)

    # The torchscript=True flag is used to ensure the model outputs are tuples
    # instead of ModelOutput (which causes JIT errors).
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, torchscript=True)

    # %%
    text = "HF models run perfectly with OpenVINO!"

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0]
    scores = torch.softmax(scores, dim=0).numpy(force=True)

    def print_prediction(scores):
        for i, descending_index in enumerate(scores.argsort()[::-1]):
            label = model.config.id2label[descending_index]
            score = np.round(float(scores[descending_index]), 4)
            print(f"{i+1}) {label} {score}")

    print_prediction(scores)

    core = ov.Core()
    save_model_path = base_model_dir / 'model.xml'

    if not save_model_path.exists():
        ov_model = ov.convert_model(model, example_input=dict(encoded_input))
        ov.save_model(ov_model, save_model_path)

    compiled_model = core.compile_model(save_model_path, device)

    # Compiled model call is performed using the same parameters as for the original model
    scores_ov = compiled_model(encoded_input.data)[0]

    scores_ov = torch.softmax(torch.tensor(scores_ov[0]), dim=0).detach().numpy()

    print_prediction(scores_ov)

    # using opimum-intel

    model = OVModelForSequenceClassification.from_pretrained(MODEL, export=True, device=device)
    optimum_model_path = base_model_dir / 'optimum_model'
    model.save_pretrained(optimum_model_path)

    output = subprocess.check_output(f"optimum-cli export openvino --model {MODEL} --task text-classification --fp16 models/optimum_model/fp16 ", shell=True)
    print(output.decode('utf-8'))

    model = OVModelForSequenceClassification.from_pretrained("models/optimum_model/fp16", device=device)

    output = model(**encoded_input)
    scores = output[0][0]
    scores = torch.softmax(scores, dim=0).numpy(force=True)

    print_prediction(scores)

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