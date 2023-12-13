#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is the notebook that was the original source of this test:
# NB_Source: 113-image-classification-quantization.ipynb

# Put a short description of the test here for the chart display,
# it must be less than 50 characters or it will be truncated:
# Test_Slug_Line: Image Quantization, NNCF

from pathlib import Path
import sys
import subprocess
import argparse
import openvino as ov
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import nncf
from tqdm.notebook import tqdm
import numpy as np

# place the base type of device [CPU, GPU, NPU] as string in UNSUPPORTED_DEVICES list
# if the device is not supported for this script
UNSUPPORTED_DEVICES = []

def main(device):
    data_dir = Path(__file__).resolve().parents[1].joinpath('data')
    base_model_dir = Path(__file__).resolve().parents[1].joinpath('models')
    model_repo = 'pytorch-cifar-models'

    data_dir.mkdir(exist_ok=True)
    base_model_dir.mkdir(exist_ok=True)

    if not Path(model_repo).exists():
        cmd = 'git clone https://github.com/chenyaofo/pytorch-cifar-models.git'
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        
    sys.path.append(model_repo)

    from pytorch_cifar_models import cifar10_mobilenetv2_x1_0

    model = cifar10_mobilenetv2_x1_0(pretrained=True)

    model.eval()

    ov_model = ov.convert_model(model, input=[1,3,32,32])

    ov.save_model(ov_model, base_model_dir / "mobilenet_v2.xml") 

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    dataset = CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    def transform_fn(data_item):
        image_tensor = data_item[0]
        return image_tensor.numpy()

    quantization_dataset = nncf.Dataset(val_loader, transform_fn)
    quant_ov_model = nncf.quantize(ov_model, quantization_dataset)
    ov.save_model(quant_ov_model, base_model_dir / "quantized_mobilenet_v2.xml")

    def test_accuracy(ov_model, data_loader):
        correct = 0
        total = 0
        for (batch_imgs, batch_labels) in tqdm(data_loader):
            result = ov_model(batch_imgs)[0]
            top_label = np.argmax(result)
            correct += top_label == batch_labels.numpy()
            total += 1
        return correct / total

    core = ov.Core()
    compiled_model = core.compile_model(ov_model, device)
    optimized_compiled_model = core.compile_model(quant_ov_model, device)

    orig_accuracy = test_accuracy(compiled_model, val_loader)
    optimized_accuracy = test_accuracy(optimized_compiled_model, val_loader)
    print(f"Accuracy of the original model: {orig_accuracy[0] * 100 :.2f}%")
    print(f"Accuracy of the optimized model: {optimized_accuracy[0] * 100 :.2f}%")

    # Define all possible labels from the CIFAR10 dataset
    labels_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    all_pictures = []
    all_labels = []

    # Get all pictures and their labels.
    for i, batch in enumerate(val_loader):
        all_pictures.append(batch[0].numpy())
        all_labels.append(batch[1].item())

    def infer_on_pictures(model, indexes: list, all_pictures=all_pictures):
        """ Inference model on a few pictures.
        :param net: model on which do inference
        :param indexes: list of indexes 
        """
        output_key = model.output(0)
        predicted_labels = []
        for idx in indexes:
            assert idx < 10000, 'Cannot get such index, there are only 10000'
            result = model(all_pictures[idx])[output_key]
            result = labels_names[np.argmax(result[0])]
            predicted_labels.append(result)
        return predicted_labels

    indexes_to_infer = [7, 12, 15, 20]  # To plot, specify 4 indexes.

    results_float = infer_on_pictures(compiled_model, indexes_to_infer)
    results_quanized = infer_on_pictures(optimized_compiled_model, indexes_to_infer)

    print(f"Labels for picture from float model : {results_float}.")
    print(f"Labels for picture from quantized model : {results_quanized}.")

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


