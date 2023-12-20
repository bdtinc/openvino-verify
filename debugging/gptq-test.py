# Description: Test script for GPTQ model
# This code came from the Note On GPTQ Models in the
# Weight Compression draft document. 
# pip install optimum[openvino] auto-gptq was run per
# the instructions in the draft document.
#
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

# Load model from Hugging Face already optimized with GPTQ
model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
model = OVModelForCausalLM.from_pretrained(model_id, export=True)

print ("model loaded")
# Inference
tokenizer = AutoTokenizer.from_pretrained(model_id)
print ("tokenizer loaded")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print ("pipeline loaded")
phrase = "The weather is"
results = pipe(phrase)
print(results)