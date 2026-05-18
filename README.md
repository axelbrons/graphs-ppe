# Graph-Aware Transformers for Smart Contract Vulnerability Detection

This repository contains the official implementation and research assets for the paper: **"Graph-Aware Transformers for Smart Contract Vulnerability Detection"**. The full paper is available at [docs/07_Papier_Scientifique_Valorisation.pdf](docs/07_Papier_Scientifique_Valorisation.pdf).

## Abstract

Smart contracts manage billions of dollars in decentralized ecosystems, yet their immutable nature makes them highly susceptible to costly, unpatchable vulnerabilities. Traditional automated auditing tools often rely on rigid, expert-defined static rules that fail to generalize, while modern Large Language Models (LLMs) frequently exhibit precision deficits in deterministic logic.

We propose a novel deep learning framework for Solidity vulnerability detection leveraging **GraphCodeBERT**, a graph-aware pre-trained Transformer. By integrating raw source code tokens with underlying Data Flow Graphs (DFGs) through a graph-guided masked attention mechanism, our approach explicitly learns anomalous data dependencies while preserving rich contextual semantics. Our model significantly outperforms state-of-the-art zero-shot LLMs. In a strictly balanced binary classification setting, our fine-tuned architecture achieves an **87% F1-score** and a **91% recall rate**.

## Key Features

- **Graph-Aware Architecture**: Utilizes GraphCodeBERT to capture both sequential semantics and structural data flow dependencies.
- **Multi-Granularity Classification**: Supports 9-class, 7-class, and binary classification tasks.
- **Robust Preprocessing**: Custom pipeline for Solidity parsing, Abstract Syntax Tree (AST) extraction, and DFG generation using Tree-sitter.
- **High Recall Performance**: Optimized to minimize False Negatives, critical for security-sensitive applications.

## Repository Structure

```text
├── data/               # Curated datasets (CSV/Parquet) and class weights
├── docs/               # Scientific paper, poster, and final presentation
├── img/                # Results visualizations and confusion matrices
├── refs/               # Reference literature and related research
└── src/                # Core implementation
    ├── config.py       # Hyperparameters and path configurations
    ├── dataset.py      # Solidity dataset handling and DFG extraction
    ├── model.py        # GraphCodeBERT classifier architecture
    ├── train.py        # Model training and fine-tuning script
    ├── test.py         # Evaluation and metrics generation
    └── data_prep.py    # Data cleaning and preprocessing pipelines
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/axelbrons/gat-smart-contracts
   cd gat-smart-contracts
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (optional, for fetching new contracts):
   Create a `.env` file with your `GITHUB_TOKEN`.

## Usage

### Data Preparation

To prepare the dataset for training:
```bash
python src/data_prep_binary.py
```

### Training

To fine-tune the GraphCodeBERT model:
```bash
python src/train.py
```
Configuration settings such as batch size, learning rate, and epochs can be modified in `src/config.py`.

### Evaluation

To evaluate the model on the test set and generate performance metrics:
```bash
python src/test.py
```

## Experimental Results

Our model was evaluated against several baseline LLMs (Llama 3, DeepSeek-Coder, etc.). The Graph-Aware Transformer demonstrates superior performance in identifying complex vulnerability patterns:

| Dataset Setup | Accuracy | Macro F1 |
|---------------|----------|----------|
| 9-Class       | 0.60     | 0.54     |
| 7-Class       | 0.63     | 0.63     |
| **Binary**    | **0.87** | **0.87** |

The binary classifier achieves a **91% recall rate** for vulnerable contracts, ensuring high reliability in detecting potential exploits.

## Citation

If you use this work in your research, please refer to the documentation provided in the `docs/` folder:

```text
Brons, A., et al. (2026). Graph-Aware Transformers for Smart Contract Vulnerability Detection.
```

## Contact

Correspondence to: Axel Brons <axel.brons@edu.ece.fr>
