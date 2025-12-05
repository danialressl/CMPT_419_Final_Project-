# Influence-Based Sample Valuation for Chest X-Ray Classification

CMPT 419 Final Project — Danial Ressl

This repository contains all code and materials for the CMPT 419 final project on influence-based data valuation for chest X-ray classification.

**Project Overview**

This project investigates how individual samples in the NIH ChestXray14 dataset contribute to model performance. A DenseNet-121 model is trained with focal loss, label smoothing, and AdamW optimization. Per-sample influence is computed with TracIn, and two data-centric experiments—removing and selecting samples based on influence—are performed. The results uncover substantial dataset redundancy and compressibility.

**Dataset**

To download NIH ChestXray14, follow instructions from Kaggle:

[https://www.kaggle.com/datasets/nih-chest-xray/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)

Place the dataset in:

data/

The repository expects the following structure:

data/images/

data/Data_Entry_2017.csv

**Setup**

Install dependencies:

pip install -r requirements.txt

**Running Experiments**

Baseline training:

python src/baseline.py --epochs 5 --loss focal --backbone densenet121

Influence scoring:

python src/tracin_influence.py --checkpoints ./checkpoints --output results/influence.csv

Subset experiments:

python src/subset_experiments.py --mode remove

python src/subset_experiments.py --mode select

**Results**

Final baseline metrics (DenseNet-121, focal loss):

Macro AUROC: 0.8139

Macro F1: 0.3502

ECE: 0.2348

Influence values ranged from –0.00246 to 0.32580.

Complete results (CSV files) are in the /results/ directory.
