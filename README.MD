# PubMedQA LoRA Fine-Tuning

Parameter-efficient fine-tuning of **GPT-Neo (125M)** on the **PubMedQA** biomedical question-answering dataset using **LoRA adapters (PEFT)**. Includes a demonstration of adapter-based unlearning via adapter detachment.

---

## About the Project

Large language models require significant compute for full fine-tuning. This project applies **LoRA (Low-Rank Adaptation)** — a parameter-efficient technique that freezes the base model weights and inserts trainable low-rank adapter matrices — to adapt GPT-Neo for a domain-specific biomedical classification task with ~99% fewer trainable parameters than full fine-tuning.

**Task:** Three-class classification — given a biomedical research question and a context passage, predict `yes`, `no`, or `maybe`.

**Dataset:** [PubMedQA](https://pubmedqa.github.io/) — a biomedical QA dataset of 1,000 expert-labeled and 61,400 unlabeled instances derived from PubMed abstracts.

---

## What This Project Does

### 1. LoRA Fine-Tuning
- Loads GPT-Neo (125M) from HuggingFace Hub
- Applies LoRA adapters via the `peft` library — only the adapter weights are trained
- Trains on PubMedQA using HuggingFace `Trainer`
- Achieves ~99% reduction in trainable parameters compared to full fine-tuning

### 2. Evaluation
- Evaluates the fine-tuned model on the PubMedQA test split
- Reports accuracy on the yes/no/maybe classification task using the `evaluate` library

### 3. Adapter-Based Unlearning Demo
- Demonstrates that LoRA adapters can be detached from the base model
- After detachment, the model reverts to its pre-fine-tuning behavior
- Shows that domain adaptation via LoRA is modular and reversible — without modifying base model weights

---

## Key Results

| | Value |
|---|---|
| Base model | GPT-Neo 125M |
| Trainable parameters (LoRA) | ~1% of total |
| Parameter reduction | ~99% vs. full fine-tuning |
| Task | 3-class classification (yes / no / maybe) |
| Dataset | PubMedQA (labeled split) |

---


## Project Structure

```
pubmedqa-lora/
├── PubMedQA_BERT_and_GPT_NEO.ipynb   # main notebook — training, eval, unlearning
├── requirements.txt                   # core dependencies (see note below)
└── README.MD
```

> **Note on `requirements.txt`:** The included file is a full Google Colab environment export and contains many unrelated packages. Use the minimal install below instead.

---

## Setup and Usage

### Option 1: Google Colab (Recommended and tested)

1. Open `PubMedQA_BERT_and_GPT_NEO.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (T4 or better)
3. Run all cells — dependencies install automatically via `pip` in the notebook

### Option 2: Local

**Requirements:** Python 3.10+, CUDA-capable GPU recommended (CPU is slow for training)

```bash
pip install transformers datasets peft accelerate evaluate torch
```

Then open and run the notebook in Jupyter:

```bash
jupyter notebook PubMedQA_BERT_and_GPT_NEO.ipynb
```

---

## Core Dependencies

| Package | Purpose |
|---|---|
| `transformers` | GPT-Neo model and tokenizer |
| `peft` | LoRA adapter implementation |
| `datasets` | PubMedQA dataset loading |
| `accelerate` | Training acceleration |
| `evaluate` | Accuracy metric |
| `torch` | Deep learning backend |

---

## Background: What is LoRA?

LoRA (Hu et al., 2021) freezes pre-trained model weights and injects trainable rank-decomposition matrices into each layer of the transformer. For a weight matrix **W**, instead of updating **W** directly, LoRA learns two small matrices **A** and **B** such that the update is **ΔW = BA**, where the rank of **ΔW** is much smaller than the original matrix dimensions.

This reduces the number of trainable parameters from hundreds of millions to a few thousand, enabling fine-tuning on consumer hardware and making the adaptation modular — adapters can be attached, detached, or swapped without changing the base model.

**Reference:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)..

---

## Related Work at UAB

This project was developed during a Graduate Research Fellowship at the University of Alabama at Birmingham at early 2024, where it was used to explore parameter-efficient adaptation of language models for a task and adapter-based unlearning as a mechanism for modular knowledge removal.
