# AI Model Trainer

A lightweight command-line toolkit for training, fine-tuning, merging, exporting, and testing Large Language Models (LLMs) using LoRA/QLoRA.

This project is designed for developers, researchers, and security professionals who want an easy-to-use training pipeline without relying on heavy web interfaces.

---

## Features

- LoRA & QLoRA Fine-tuning
- Configuration-based training
- Dataset preparation utilities
- Model merging
- Model exporting
- Inference support
- Training logs
- Checkpoint management
- Modular Python architecture
- CLI-based workflow

---

## Project Structure

```
AI-Model-Trainer/
├── configs/
├── datasets/
├── exports/
├── logs/
├── models/
│   ├── adapters/
│   ├── base/
│   ├── checkpoints/
│   └── merged/
├── outputs/
├── scripts/
├── src/
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Squzu/AI-Model-Trainer.git
cd AI-Model-Trainer
```

Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

All settings are stored inside the `configs/` directory.

- `training.yaml`
- `model.yaml`
- `dataset.yaml`
- `inference.yaml`

Modify these files before starting training.

---

## Training

```bash
python scripts/train.py
```

---

## Dataset Preparation

```bash
python scripts/prepare_dataset.py
```

---

## Merge LoRA Adapter

```bash
python scripts/merge_lora.py
```

---

## Export Model

```bash
python scripts/export.py
```

---

## Run Tests

```bash
python scripts/test.py
```

---

## Folder Description

| Folder | Purpose |
|----------|----------|
| configs | Configuration files |
| datasets | Training datasets |
| models | Base models, adapters, checkpoints |
| outputs | Training outputs |
| exports | Exported models |
| logs | Training logs |
| scripts | CLI scripts |
| src | Core Python modules |

---

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- NVIDIA GPU with sufficient VRAM
- Linux (recommended)

---

## Roadmap

- [x] CLI Training Pipeline
- [x] Dataset Preparation
- [x] LoRA Training
- [x] QLoRA Support
- [x] Model Merge
- [x] Model Export
- [ ] Multi-GPU Training
- [ ] Distributed Training
- [ ] Automatic Dataset Validation
- [ ] Hugging Face Hub Integration
- [ ] Docker Support
- [ ] Web Dashboard

---

## Contributing

Contributions are welcome.

Feel free to submit issues, feature requests, or pull requests.

---

## License

This project is licensed under the MIT License.

---

## Author

**Shridhar Kamath**

Cybersecurity Engineer | SIEM | Detection Engineering | AI Security | LLM Research
