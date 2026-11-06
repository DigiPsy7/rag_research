
import torch
import gc
from omegaconf import OmegaConf

from code.utils.utils import get_device
from code.utils.load_data import load_data
from code.finetune.model import Trainer
from code.finetune.batch_collator import CollateWraperGenerative

print("Starting one-batch training test...")

configs = OmegaConf.load("./code/finetune/configs.yaml")

device = get_device(configs)

print("Device:", device)

# Load the existing dataset
train_set, val_set, test_set = load_data(configs)

print("Training samples:", len(train_set))

# Use only one sample for this test
train_sample = train_set[:1]

# Load model and LoRA
trainer = Trainer(configs, device, num_training_steps=1)

print("Model loaded successfully.")

# Prepare one batch
collator = CollateWraperGenerative(
    trainer.tokenizer,
    device,
    configs,
    None
)

batch = collator(train_sample)

print("Batch prepared.")

# One forward and backward pass
trainer.set_train_mode()

logs = trainer.train_step(batch)

print("Training step completed.")
print("Loss:", logs["loss"].item())

print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated(0) / 1024**3, 2),
    "GB"
)

print(
    "GPU peak memory:",
    round(torch.cuda.max_memory_allocated(0) / 1024**3, 2),
    "GB"
)

del trainer, batch, collator
gc.collect()
torch.cuda.empty_cache()

print("ONE-BATCH TRAINING TEST SUCCESSFUL")