#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is the notebook that was the original source of this test:
# NB_Source: 267-distil-whisper-asr.ipynb

# Put a short description of the test here for the chart display,
# it must be less than 50 characters or it will be truncated:
# Test_Slug_Line: Distil Whisper ASR

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
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
    data_dir = Path(__file__).resolve().parents[0].joinpath('data')
    model_dir = Path(__file__).resolve().parents[0].joinpath('models')

    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    distil_model_id = "distil-whisper/distil-large-v2"

    processor = AutoProcessor.from_pretrained(distil_model_id)

    pt_distil_model = AutoModelForSpeechSeq2Seq.from_pretrained(distil_model_id)
    pt_distil_model.eval();


    from datasets import load_dataset

    def extract_input_features(sample):
        input_features = processor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt",
        ).input_features
        return input_features

    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = dataset[0]
    input_features = extract_input_features(sample)

    
#    import IPython.display as ipd

    predicted_ids = pt_distil_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

#    display(ipd.Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"]))
    print(f"Reference: {sample['text']}")
    print(f"Result: {transcription[0]}")

    from optimum.intel.openvino import OVModelForSpeechSeq2Seq

    distil_model_path = Path(distil_model_id.split("/")[-1])

    if not distil_model_path.exists():
        ov_distil_model = OVModelForSpeechSeq2Seq.from_pretrained(
            distil_model_id, export=True, compile=False
        )
        ov_distil_model.half()
        ov_distil_model.save_pretrained(distil_model_path)
    else:
        ov_distil_model = OVModelForSpeechSeq2Seq.from_pretrained(
            distil_model_path, compile=False
        )

    

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