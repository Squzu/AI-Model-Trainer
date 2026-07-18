<div align="center">

# AI Model Trainer

**A lightweight command-line toolkit for fine-tuning, merging, exporting, and testing Large Language Models using LoRA and QLoRA — built to run on modest, consumer-grade GPUs.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.46.3-FFD21E?logo=huggingface&logoColor=black)
![Ollama](https://img.shields.io/badge/Ollama-Compatible-000000?logo=ollama&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20(WSL)-informational)

</div>

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Who This Is For](#who-this-is-for)
3. [How It Works, In Plain Terms](#how-it-works-in-plain-terms)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Configuration Files Explained](#configuration-files-explained)
8. [Preparing Your Dataset](#preparing-your-dataset)
9. [Converting Your Own Dataset (Prompt for ChatGPT/Claude)](#converting-your-own-dataset-prompt-for-chatgptclaude)
10. [Step-by-Step Usage](#step-by-step-usage)
11. [Changing the Base Model](#changing-the-base-model)
12. [Changing GPU / Adjusting for Your VRAM](#changing-gpu--adjusting-for-your-vram)
13. [Exporting Your Model](#exporting-your-model)
14. [Troubleshooting](#troubleshooting)
15. [Roadmap](#roadmap)
16. [License](#license)
17. [Credits](#credits)

---

## What This Project Does

This toolkit lets you take an existing open-source language model (from Hugging Face) and teach it your own data, without needing a data center. It uses a technique called **LoRA (Low-Rank Adaptation)** combined with **4-bit quantization (QLoRA)**, which is what makes it possible to fine-tune a real language model on a normal gaming GPU with as little as 4 GB of VRAM.

Once trained, you can:

- Merge your fine-tuned adapter back into the base model
- Export it as a standalone Hugging Face model
- Convert it to GGUF format for fast local inference
- Deploy it directly to Ollama
- Test it interactively from the command line

Everything is controlled through simple YAML configuration files and single-purpose Python scripts — there is no web interface, no notebook required, and no manual editing of Python code needed for normal use.

---

## Who This Is For

You do **not** need to know how to code to use this project. You need to be comfortable with:

- Opening a terminal / command prompt
- Copy-pasting commands
- Editing plain text files (YAML configuration files)

Anywhere this README says "run this command," it means: open your terminal, make sure you are inside the project folder, paste the command, and press Enter.

---

## How It Works, In Plain Terms

Think of it as five stages, each handled by one script:

| Stage | Script | What it does |
|---|---|---|
| 1. Prepare | `scripts/prepare_dataset.py` | Turns your raw data into the training format the model expects |
| 2. Train | `scripts/train.py` | Fine-tunes the base model on your data using LoRA |
| 3. Merge | `scripts/merge_lora.py` | Combines your trained adapter with the base model into one model |
| 4. Export | `scripts/export.py` | Converts the merged model into the format you want to use it in |
| 5. Test | `scripts/test.py` | Lets you chat with your fine-tuned model from the terminal |

You run these in order, once per training cycle.

---

## Project Structure

```
AI-Model-Trainer/
├── configs/                 Configuration files (edit these, not the code)
│   ├── model.yaml           Which model to use, and quantization settings
│   ├── training.yaml        Training hyperparameters
│   ├── dataset.yaml         Dataset file paths and split settings
│   └── inference.yaml       Settings used when testing/chatting with the model
├── datasets/
│   ├── raw/                 Put your original, unprocessed dataset here
│   └── processed/           Auto-generated training/validation files go here
├── models/
│   ├── base/                Downloaded base model gets cached here
│   ├── adapters/            Your trained LoRA adapter is saved here
│   ├── checkpoints/         Training checkpoints (for resuming interrupted runs)
│   └── merged/              The final merged model after merge_lora.py
├── exports/                 Final exported models (Hugging Face / GGUF / Ollama)
├── logs/                    Timestamped log files from every run
├── outputs/                 Training run outputs (Trainer state, TensorBoard logs)
├── scripts/                 The commands you actually run
├── src/                     Core Python modules (you shouldn't need to edit these)
├── requirements.txt         Exact package versions this project needs
└── README.md                This file
```

---

## Requirements

**Hardware**

- An NVIDIA GPU with CUDA support (this project is tuned for cards with as little as 4 GB VRAM, such as the GTX 1050 Ti, but works better with more)
- At least 16 GB of system RAM recommended
- A few GB of free disk space per model you download

**Software**

- Python 3.10 or newer
- Git
- An NVIDIA driver with a reasonably recent CUDA toolkit (CUDA 11.8+ is enough)
- Linux is recommended. On Windows, use **WSL2** (Windows Subsystem for Linux) — running CUDA training natively on Windows is possible but far more failure-prone.

To check what you currently have installed, run:

```bash
python scripts/setup.py
```

This prints your Python version, whether Git/CMake/G++ are available, your GPU name and VRAM, and whether Ollama and llama.cpp are installed.

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/squzu/AI-Model-Trainer.git
cd AI-Model-Trainer
```

**2. Create a virtual environment** (keeps this project's packages separate from the rest of your system)

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows (WSL): same command
```

You'll know it worked because your terminal prompt will now start with `(.venv)`.

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

This will take a few minutes — it downloads PyTorch and several large libraries.

**4. Verify your setup**

```bash
python scripts/setup.py
```

Confirm your GPU is detected before moving on.

---

## Configuration Files Explained

All settings live in `configs/` as plain YAML files. You will mainly touch `model.yaml`, `training.yaml`, and `dataset.yaml`.

### `configs/model.yaml`

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"   # Hugging Face model ID
  cache_dir: "./models/base"            # where the downloaded model is stored
  load_in_4bit: true                    # keep this true on small GPUs
  bnb_4bit_quant_type: "nf4"             # quantization type, leave as-is
  bnb_4bit_compute_dtype: "float16"      # use float16 on GTX 10-series and older
  bnb_4bit_use_double_quant: true
  trust_remote_code: true

lora:
  r: 4                # LoRA rank — higher = more capacity, more VRAM
  alpha: 8
  dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules:      # which layers LoRA is applied to
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### `configs/training.yaml`

```yaml
training:
  output_dir: "./outputs"
  adapter_dir: "./models/adapters"
  epochs: 3
  batch_size: 1                         # keep at 1 on low-VRAM GPUs
  gradient_accumulation_steps: 32       # simulates a larger batch size
  learning_rate: 0.0002
  warmup_ratio: 0.03
  weight_decay: 0.01
  logging_steps: 10
  save_steps: 100
  save_total_limit: 2
  fp16: true                            # keep true on GTX 10/16-series
  bf16: false                           # only enable on RTX 20-series or newer
  max_seq_length: 128                   # increase only if you have VRAM to spare
  optim: "paged_adamw_8bit"
  lr_scheduler: "cosine"
  seed: 42
```

### `configs/dataset.yaml`

```yaml
dataset:
  train_file: "./datasets/processed/train.json"
  validation_file: "./datasets/processed/validation.json"
  text_column: "text"
  shuffle: true
  validation_split: 0.1     # 10% of data held out for validation
```

### `configs/inference.yaml`

Controls text generation settings used by `scripts/test.py` (temperature, max tokens, etc.) — safe to leave at defaults.

---

## Preparing Your Dataset

`scripts/prepare_dataset.py` expects your raw data at:

```
datasets/raw/train.json
```

...as a JSON array where each entry has exactly three fields:

```json
[
  {
    "instruction": "Summarize the following text.",
    "input": "The quick brown fox jumps over the lazy dog.",
    "output": "A fox jumps over a dog."
  },
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "Paris."
  }
]
```

- `instruction` — what you're asking the model to do
- `input` — optional extra context (leave as an empty string `""` if not needed)
- `output` — the answer the model should learn to produce

Once your file is in that format, run:

```bash
python scripts/prepare_dataset.py
```

This automatically builds the prompt format the model needs, shuffles your data, splits it into training and validation sets, and writes both to `datasets/processed/`.

---

## Converting Your Own Dataset (Prompt for ChatGPT/Claude)

If your existing data is in a different shape (a CSV, a spreadsheet export, a chat log, a plain list of Q&A pairs, etc.), you don't need to reformat it by hand. Copy the prompt below into ChatGPT or Claude, fill in the blanks describing your actual file, and it will write you a ready-to-run Python script that converts your data into the format this project needs.

```
I have a dataset I want to convert into a specific JSON format for fine-tuning
a language model. Write me a complete, runnable Python script that does this
conversion.

My current dataset format is:
[DESCRIBE YOUR FILE HERE — e.g. "a CSV file with columns 'question' and 'answer'"
 or "a folder of .txt files, one conversation per file" — and paste 2-3 real
 example rows/lines if you can]

My dataset is located at: [PASTE THE FILE PATH, e.g. "datasets/raw/my_data.csv"]

I need the script to output a JSON file at datasets/raw/train.json as a JSON
array, where every entry looks exactly like this:

{
  "instruction": "<the task or question>",
  "input": "<optional extra context, or an empty string if none>",
  "output": "<the expected answer or response>"
}

Requirements for the script:
- Read my source file from the path I gave you.
- Map my existing fields into "instruction", "input", and "output" (explain
  your mapping choices in a comment if it's not obvious).
- If a field doesn't exist in my data, fill "input" with an empty string
  rather than leaving it out.
- Skip or clearly log any rows that are missing required data instead of
  crashing.
- Save the final result as valid JSON (a list of objects) at
  datasets/raw/train.json, using UTF-8 encoding.
- Use only Python's standard library plus pandas if needed for CSV/Excel
  files — nothing else.
- Add a short print statement at the end showing how many records were
  converted.
```

Once the script finishes and you have `datasets/raw/train.json` in the right format, continue with `python scripts/prepare_dataset.py` as described above.

---

## Step-by-Step Usage

Run these from the project's root folder, in this order, with your virtual environment activated.

**1. Prepare your dataset**

```bash
python scripts/prepare_dataset.py
```

**2. Train**

```bash
python scripts/train.py
```

This downloads the base model on first run (cached afterward), applies 4-bit quantization, attaches a LoRA adapter, and starts training. Progress is printed to the terminal and saved to `logs/`. If training is interrupted, running this command again automatically resumes from the last saved checkpoint instead of starting over.

**3. Merge the trained adapter into the base model**

```bash
python scripts/merge_lora.py
```

This step runs on CPU intentionally — it needs the full model in memory at once, which typically won't fit in 4 GB of VRAM. It only needs to run once per training run, and does not require a GPU.

**4. Export**

```bash
python scripts/export.py --target hf       # Hugging Face format
python scripts/export.py --target gguf      # GGUF format (for llama.cpp)
python scripts/export.py --target ollama    # Ollama-ready Modelfile
```

**5. Test your model**

```bash
python scripts/test.py
```

Type a question and press Enter. Type `exit` to quit.

---

## Changing the Base Model

To use a different model, open `configs/model.yaml` and change:

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
```

...to any model ID from Hugging Face, for example:

```yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
```

Notes:

- The model must be a causal language model (the kind used for text generation, not e.g. BERT-style encoders).
- Some models (like Llama) require you to accept a license on Hugging Face and log in locally first, using:
  ```bash
  huggingface-cli login
  ```
- Bigger models need more VRAM even at 4-bit. As a rough guide: a 1–3B parameter model fits comfortably in 4 GB VRAM at 4-bit; 7B models generally need at least 6–8 GB VRAM at 4-bit, and will be slow or may not fit at all on a 4 GB card.
- If you switch models, delete or rename the old contents of `models/adapters/` first — an adapter trained for one model's architecture will not load correctly on a different model.

---

## Changing GPU / Adjusting for Your VRAM

The default settings in `configs/training.yaml` are tuned for a 4 GB card (like a GTX 1050 Ti). If you move to a different GPU, here's what to adjust:

| Your VRAM | `batch_size` | `gradient_accumulation_steps` | `lora.r` | `max_seq_length` | `bf16` |
|---|---|---|---|---|---|
| 4 GB (GTX 1050 Ti, GTX 1650) | 1 | 32 | 4 | 128 | false |
| 6–8 GB (GTX 1660, RTX 2060/3050/3060) | 1–2 | 16 | 8 | 256 | true on RTX 20-series+ |
| 8–12 GB (RTX 3060 Ti/3070/3080) | 2–4 | 8–16 | 16 | 512 | true |
| 16 GB+ (RTX 4080/4090, data-center cards) | 4–8 | 4–8 | 32 | 1024+ | true |

General rules of thumb:

- **Increase `batch_size`** before decreasing `gradient_accumulation_steps` — keep the product of the two roughly similar if you want the same effective training behavior.
- **`bf16` (bfloat16)** is only supported on NVIDIA Turing (RTX 20-series) and newer. GTX 10-series and 16-series GPUs (Pascal/Turing without Tensor Cores for bf16) must keep `fp16: true` and `bf16: false`.
- **Increasing `lora.r`** increases how much the adapter can learn, at the cost of more VRAM and slightly slower training. Start low and increase only if you have headroom.
- **`max_seq_length`** has a large effect on VRAM usage — doubling it roughly doubles activation memory. Only raise it if your GPU has room.
- If you hit an "out of memory" (OOM) error, in order of preference: lower `max_seq_length`, lower `lora.r`, make sure `batch_size` is `1`, then raise `gradient_accumulation_steps` to compensate.

---

## Exporting Your Model

- **Hugging Face format** (`--target hf`) — a plain folder you can load with `AutoModelForCausalLM.from_pretrained(...)`, or upload to the Hugging Face Hub.
- **GGUF** (`--target gguf`) — required for running the model with `llama.cpp` or many local inference apps. You'll need `llama.cpp` cloned into `tools/llama.cpp` first:
  ```bash
  git clone https://github.com/ggerganov/llama.cpp tools/llama.cpp
  ```
- **Ollama** (`--target ollama`) — generates a `Modelfile` pointing at your GGUF export. Requires the GGUF export step to be run first, and [Ollama](https://ollama.com) installed locally. After exporting, run:
  ```bash
  ollama create ai-model-trainer -f exports/ollama/Modelfile
  ollama run ai-model-trainer
  ```

---

## Troubleshooting

**`ImportError` when running `merge_lora.py`**
Make sure you're running the version of this repo with `merge_lora()` implemented in `src/exporter.py` — an earlier version of this project shipped without it.

**`CUDA out of memory`**
See the [VRAM table above](#changing-gpu--adjusting-for-your-vram). Lower `max_seq_length` first, then `lora.r`, and make sure `batch_size` is `1`.

**`bitsandbytes` fails to load / no GPU support detected**
Confirm your NVIDIA driver is installed and `nvidia-smi` works in your terminal. Run `python scripts/setup.py` to check. Reinstall with the exact pinned version in `requirements.txt` rather than the latest release, since newer releases can drop support for older hardware or require a newer PyTorch than what's pinned here.

**Training is extremely slow**
This is expected on older/lower-VRAM GPUs — 4-bit quantization trades speed for fitting the model into limited memory. Reducing `max_seq_length` and dataset size will speed things up.

**Model gives nonsense or repeats itself when testing**
Usually means training needs more epochs, a larger/cleaner dataset, or the learning rate needs tuning. Try increasing `epochs` in `training.yaml` first.

**Export to GGUF fails immediately**
You likely haven't cloned `llama.cpp` into `tools/llama.cpp` yet — see the [Exporting](#exporting-your-model) section above.

---

## Roadmap

- [x] CLI training pipeline
- [x] Dataset preparation
- [x] LoRA training
- [x] QLoRA support
- [x] Model merge
- [x] Model export (Hugging Face / GGUF / Ollama)
- [x] Checkpoint resume support
- [ ] Multi-GPU training
- [ ] Distributed training
- [ ] Automatic dataset validation
- [ ] Hugging Face Hub upload integration
- [ ] Docker support
- [ ] Web dashboard

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Credits

<div align="center">

**Made by Shridhar Kamath + ChatGPT + Claude**

</div>
