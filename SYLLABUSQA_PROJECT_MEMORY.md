# SYLLABUSQA DISSERTATION — MASTER PROJECT MEMORY

**Last updated:** 5 November 2026

## 1. Project Overview

Developing a SyllabusQA dissertation project based on the SyllabusQA research paper.

The project uses Llama-2-7B-Chat fine-tuned using QLoRA (Quantized Low-Rank Adaptation).

The implementation is based on the original project structure, with modifications to support local Windows training and an NVIDIA RTX 3070 Ti Laptop GPU with 8 GB VRAM.

Current experimental approach:

* Standard Supervised Fine-Tuning (SFT)
* QLoRA fine-tuning
* RAG disabled for the current experiment
* Evaluation using ROUGE-L F1

## 2. Hardware and Environment

* Operating system: Windows 11
* Terminal: PowerShell
* GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
* GPU VRAM: 8 GB
* System RAM: Approximately 16 GB
* CUDA available: True
* PyTorch: 2.13.0+cu126

### Python environment

* Python virtual environment: `.venv`
* Transformers: 5.22.1
* PEFT: 0.21.2
* BitsAndBytes: 0.50.2
* Accelerate: 1.15.0

## 3. Project Directory

`D:\lachu-research\SyllabusQA-Dissertation-main\SyllabusQA-Dissertation-main`

Important directories:

* `code/finetune/`
* `code/utils/`
* `data/dataset_split/`
* `checkpoints/`
* `results/`

## 4. Base Model

Model: `meta-llama/Llama-2-7b-chat-hf`

The complete base model is downloaded and cached locally.

The model successfully loads in 4-bit quantization.

## 5. Fine-Tuning Methodology

QLoRA configuration:

| Parameter                        | Value                                                                  |
| -------------------------------- | ---------------------------------------------------------------------- |
| Quantization                     | 4-bit                                                                  |
| Quantization type                | NF4                                                                    |
| Double quantization              | Enabled                                                                |
| Computation dtype                | FP16                                                                   |
| LoRA rank                        | 16                                                                     |
| LoRA alpha                       | 32                                                                     |
| LoRA dropout                     | 0.05                                                                   |
| Target modules                   | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head |
| Gradient checkpointing           | Enabled                                                                |
| Gradient checkpointing reentrant | False                                                                  |
| use_cache                        | False                                                                  |
| Device mapping                   | {"": 0}                                                                |

Trainable parameters verified: 40,554,496.

## 6. Dataset

The SyllabusQA dataset has been downloaded, verified and loaded successfully.

| Dataset    |   Samples |
| ---------- | --------: |
| Training   |     3,018 |
| Validation |       957 |
| Testing    |     1,103 |
| **Total**  | **5,078** |

Dataset loading has been verified through the project's data-loading pipeline.

## 7. Training Configuration

Configuration file: `code/finetune/configs.yaml`

| Parameter               | Value  |
| ----------------------- | ------ |
| Maximum sequence length | 512    |
| Training batch size     | 1      |
| Validation batch size   | 1      |
| Test batch size         | 1      |
| Epochs                  | 3      |
| Learning rate           | 1e-4   |
| Warmup steps            | 272    |
| Gradient clipping       | 1.0    |
| LR scheduler            | Cosine |
| RAG                     | False  |
| Debug                   | False  |
| W&B logging             | False  |
| Run testing             | False  |

Planned total training steps: 9,054.

## 8. Verified Accomplishments

### Environment and model verification

1. Python virtual environment configured.
2. Dependencies installed and compatibility issues resolved.
3. NumPy, Pandas and PyArrow compatibility issues fixed.
4. CUDA GPU detection successful.
5. BitsAndBytes Linear4bit imported successfully.
6. Actual 4-bit GPU computation verified.
7. Llama-2-7B loaded successfully in 4-bit.
8. Initial model loading used approximately 3.61 GB GPU memory.
9. LoRA adapters created successfully.
10. Trainable parameters verified: 40,554,496.

### Training verification

11. One-batch forward pass successful.
12. One-batch backward pass successful.
13. One-batch training loss: 4.544266700744629.
14. Peak GPU memory during the initial training-step test: 4.88 GB.
15. `model.py` updated with `dtype=torch.float16`.
16. Non-reentrant gradient checkpointing configured.
17. Updated one-batch training test successful.
18. `python -m py_compile` checks passed.

### Full training

19. Full training launched using:

`python -m code.finetune.train`

20. All three training epochs completed successfully.

21. Training execution time reported: approximately 1 hour, 41 minutes, 13 seconds.

22. Final displayed training loss: approximately 0.467.

23. Final displayed validation loss: approximately 1.95.

24. Best-validation-loss checkpoint successfully saved.

**Important:** The final displayed validation loss is not necessarily the best validation loss across all three epochs. The saved checkpoint is selected by the validation-loss comparison in `train.py`.

## 9. Saved Checkpoint

Verified checkpoint directory:

`checkpoints/sft/sft/best_val_loss/lora_model/`

Files verified:

* `adapter_config.json`
* `adapter_model.safetensors`
* `chat_template.jinja`
* `README.md`
* `tokenizer.json`
* `tokenizer_config.json`

Adapter model file size: approximately 686.6 MB.

The checkpoint contains the trained LoRA adapters rather than a complete copy of the 7B base model.

The adapter configuration confirms:

* Base model: Llama-2-7B-Chat
* LoRA rank: 16
* LoRA alpha: 32
* LoRA dropout: 0.05
* Eight target module types
* PEFT version: 0.21.2

W&B is disabled. The project uses `configs.exp_name` as the fallback experiment name.

## 10. Important Code Files

| File                              | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| `code/finetune/model.py`          | Model loading, quantization, LoRA and training |
| `code/finetune/train.py`          | Training loop and validation                   |
| `code/finetune/test.py`           | Test-set evaluation                            |
| `code/finetune/configs.yaml`      | Experiment configuration                       |
| `code/finetune/batch_collator.py` | Batch preparation                              |
| `code/utils/load_data.py`         | Dataset loading                                |
| `code/utils/utils.py`             | Utility functions and metrics                  |

## 11. Current Project Status

**FULL QLoRA FINE-TUNING: COMPLETED**

Verified status:

* [x] Environment configured
* [x] Dataset loaded
* [x] GPU verified
* [x] 4-bit quantization verified
* [x] LoRA adapters created
* [x] One-batch training verified
* [x] Full three-epoch training completed
* [x] Validation performed
* [x] Best-validation-loss checkpoint saved
* [x] Checkpoint files verified
* [x] Test script syntax checked

Remaining:

* [ ] Execute test-set evaluation.
* [ ] Record overall ROUGE-L F1.
* [ ] Record ROUGE-L F1 by question type.
* [ ] Verify prediction CSV and evaluation report.
* [ ] Compare results against the search baseline.
* [ ] Organize experimental results for dissertation.
* [ ] Prepare methodology and results chapters.

## 12. Immediate Next Actions

### Step 1: Run test-set evaluation

First inspect the evaluation configuration and checkpoint path.

The current evaluation script is:

`code/finetune/test.py`

The script loads the saved PEFT adapter, loads the test dataset and generates predictions.

### Step 2: Evaluate the trained model

The evaluation should use the saved checkpoint:

`checkpoints/sft/sft/best_val_loss/lora_model/`

The test dataset contains 1,103 samples.

### Step 3: Record metrics

Record:

* Overall ROUGE-L F1
* ROUGE-L F1 for multi factual
* ROUGE-L F1 for multi reasoning
* ROUGE-L F1 for no answer
* ROUGE-L F1 for single factual
* ROUGE-L F1 for single reasoning
* ROUGE-L F1 for summarization
* ROUGE-L F1 for yes/no

### Step 4: Compare against baseline

Previously verified search-baseline result:

Overall ROUGE-L F1: 0.12052818634788906.

This is the search-baseline result, not the fine-tuned model's result.

The comparison should use the same test dataset and evaluation methodology.

### Step 5: Dissertation documentation

Prepare:

* Experimental setup
* Hardware and software specifications
* Dataset description
* QLoRA methodology
* Training configuration
* Training and validation results
* Test evaluation results
* Baseline comparison
* Limitations and discussion

## 13. Important Research Notes

1. Do not claim that the final test evaluation is completed until the evaluation script has successfully run.
2. Do not confuse validation loss with test performance.
3. Keep the saved best-validation-loss checkpoint unchanged.
4. Preserve the original search-baseline results.
5. Record all experimental configurations for reproducibility.
6. Clearly distinguish original paper methodology from modifications required for local hardware.
7. Do not claim exact reproduction of the paper until its reported experimental settings and results have been compared.

## 14. Instructions for Future ChatGPT Sessions

Act as the technical research assistant for this SyllabusQA dissertation.

Continue from the current project state.

Do not restart installation, dataset downloading, GPU verification or one-batch training tests unless a specific issue requires it.

When troubleshooting:

* Inspect actual terminal output.
* Explain what succeeded and what failed.
* Provide PowerShell-compatible commands.
* Give exact file modifications when necessary.
* Change only what is required.
* Prioritize compatibility with the 8 GB GPU.
* Distinguish verified results from assumptions.
* Guide the project one step at a time.
* Wait for terminal output before proceeding to the next troubleshooting step.

**Current priority: Complete test-set evaluation of the successfully fine-tuned Llama-2-7B QLoRA model and compare it with the previously evaluated search baseline.**
