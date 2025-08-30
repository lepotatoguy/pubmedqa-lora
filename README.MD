# PubMedQA LoRA Fine-Tuning

This repository contains demonstration of **parameter-efficient fine-tuning (PEFT)** of **GPT-Neo (125M)** on the **PubMedQA biomedical QA dataset** using **LoRA adapters**.

## Overview
- Fine-tunes GPT-Neo for **yes/no/maybe classification** on PubMedQA.  
- Uses **LoRA adapters** (~99% parameter reduction) for efficient domain adaptation.  
- Includes a simple **unlearning demo** by detaching adapters.  

## Files
- `pubmedqa_lora.ipynb` – main notebook with training, evaluation, and unlearning demonstration.  

## Requirements

```
transformers
datasets
peft
accelerate
evaluate
torch
```

## Usage

I ran it primarily on Google Colab. So, you can download the file and run it there. Or I have added the requirements.txt that you can use to install the dependencies. 
