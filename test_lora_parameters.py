
import torch
from omegaconf import OmegaConf
from code.finetune.model import LanguageModel

configs = OmegaConf.load("./code/finetune/configs.yaml")

device = torch.device("cuda")

print("Loading quantized model with LoRA...")

model = LanguageModel(configs, device)

model.model.print_trainable_parameters()

trainable = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

total = sum(
    p.numel() for p in model.parameters()
)

print(f"Trainable parameters: {trainable:,}")
print(f"Total parameters: {total:,}")
print(f"Trainable percentage: {100 * trainable / total:.4f}%")

print("GPU memory allocated:",
      round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB")

del model
torch.cuda.empty_cache()

print("LoRA verification completed.")