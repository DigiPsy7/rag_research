
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "meta-llama/Llama-2-7b-chat-hf"

print("Starting 4-bit model loading test...")
print("GPU:", torch.cuda.get_device_name(0))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("MODEL LOADED SUCCESSFULLY")
print("GPU memory allocated:",
      round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB")

del model
torch.cuda.empty_cache()

print("Test completed.")