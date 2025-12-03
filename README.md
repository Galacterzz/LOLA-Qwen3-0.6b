# LOLA Implementation: Optimizing Headline Selection with Qwen SLMs

This Project is done as a part of EE 782 IIT Bombay - Advanced Machine Learning  
By students:  
1. Anirudha Shinde (22b2181)
2. Valerian Minz (24M1074)
3. Samart Sirsat (22b3988)

[DEMO](https://drive.google.com/file/d/1OpMPzz5hSq612xiDPXd-kbbcKk3z3Rp2/view?usp=sharing)  |  [Report](url)

## 📜 Overview

This project implements a framework inspired by the LOLA (LLM-Assisted Online Learning Algorithm) paper. The goal is to automate the selection of high-performing news headlines (maximizing Click-Through Rate, or CTR) using Small Language Models (SLMs).

We investigate two distinct approaches using the Qwen-0.6B architecture:
1. Embedding-Based Regression: Using semantic embeddings to train a regressor that predicts CTR directly.
2. Generative Fine-Tuning (LoRA): Fine-tuning the LLM to act as an "Expert Editor" that selects the best headline from a list.

## 📊 Results Summary

Despite the "Expert Editor" (Fine-Tuning) approach being more intuitive, our experiments show that the Embedding Regression model significantly outperforms the generative approach on this specific dataset.

| Method | Model | Accuracy (Hit Rate) |
|:-------------|:--------------:|--------------:|
| Fine-Tuned LLM (LoRA)       | Qwen3-0.6B-Base     | 35.70%  |
| Embedding Regression       | Qwen3-Embedding(1024) + MLP     | 42.79%  |

Note: Accuracy is defined as the percentage of times the model correctly identified the headline with the highest actual CTR from a set of options.

## 🛠️ Project Structure

The project is divided into 4 sequential notebooks:

1. lola-embeded.ipynb (Embedding Model)
  - **Data**: Loads the Upworthy dataset (17k+ A/B tests).
  - **Processing**: Preprocesses data into Train/Calibrate/Test splits.
  - **Embeddings**: Generates vector embeddings for all headlines using Qwen/Qwen3-Embedding-0.6B.
  - **Training**: Trains a Multi-Layer Perceptron (MLP) regressor to predict CTR from embeddings.
  - **Hyperparameter Tuning**: Grid search identified the best config as {'hidden_dims': [512, 256], 'lr': 0.001, 'dropout': 0.2}.

2. lola-finetune-dataset.ipynb (Data Prep)
  - Formatting: Converts the regression dataset into a conversational format for the LLM.
  - Prompt Engineering: Constructs the System and User prompts:
    "You are an editor tasked with choosing the catchier one from several drafted headlines..."
  - Output: Produces finetune_train.json, finetune_calibrate.json, and finetune_test.json.

3. qwen_finetune_lola.ipynb (Fine-Tuning)
  - Model: Qwen/Qwen3-0.6B-Base loaded in 4-bit precision (NF4).
  - Technique: LoRA (Low-Rank Adaptation) to efficiently update weights.
  - Config:
    - Rank (r): 16
    - Alpha: 32
    - Target Modules: All linear layers (q_proj, k_proj, v_proj, o_proj, etc.)
- Training: Uses Hugging Face SFTTrainer.

4. final-results-lola.ipynb (Evaluation)
  - Inference: Runs the fine-tuned model on the test set to generate predictions.
  - Comparison: Merges results from both models and calculates the final accuracy against the ground truth CSV.

## 🚀 How to Run

Prerequisites
Ensure you have a GPU available (Google Colab T4 or better recommended).
1. Install Dependencies
```
pip install -r requirements.txt
```

2. Run the Notebooks in Order
  1. Run lola-embeded.ipynb to generate the CSV data and train the regression model.
  2. Run lola-finetune-dataset.ipynb to prepare the JSON files.
  3. Run qwen_finetune_lola.ipynb to perform the LoRA fine-tuning.
  4. Run final-results-lola.ipynb to see the final comparison.

## 🧠 Methodology Details

**Why Qwen-0.6B?**
We chose the Qwen-0.6B family to explore the capabilities of Small Language Models (SLMs). While larger models (7B+) generally have higher reasoning capabilities, 0.6B allows for rapid iteration and deployment on edge devices. The performance drop compared to larger state-of-the-art models is expected but provides a valuable baseline for efficient computing.

**Dataset**
The dataset is sourced from the Upworthy Archive, containing thousands of headline A/B tests. Each test consists of multiple headline variations for the same article and the resulting click-through rate.

## 📚 References

- **Original Paper**: LOLA: LLM-Assisted Online Learning Algorithm
- **Dataset**: Upworthy LOLA Dataset on Kaggle
- **Base Model**: Qwen3-0.6B on Hugging Face
