# src/test.py
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src import config
from src.dataset import SolidityDataset
from src.model import GraphCodeBERTClassifier

def test():
    # 1. Load Data
    print(f"Loading data from {config.DATA_PATH}...")
    df = pd.read_parquet(config.DATA_PATH)
    
    # We must use the same split to get the test set
    _, temp_df = train_test_split(df, test_size=0.3, random_state=config.SEED, stratify=df['label_encoded'])
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=config.SEED, stratify=temp_df['label_encoded'])

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    test_dataset = SolidityDataset(test_df, tokenizer, config.MAX_SEQ_LEN, config.MAX_VAR_LEN)
    
    # --- OPTIMISATION 1 : DataLoader Turbo ---
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE,
        num_workers=4,       # Utilise 4 cœurs CPU en parallèle
        pin_memory=True      # Prépare la mémoire pour un transfert ultra-rapide vers le GPU
    )

    # 2. Load Model
    num_labels = df['label_encoded'].nunique()
    model = GraphCodeBERTClassifier(config.MODEL_NAME, num_labels)
    
    if not os.path.exists(config.SAVE_PATH):
        print(f"Error: Model file {config.SAVE_PATH} not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # 3. Evaluation
    all_preds = []
    all_labels = []
    test_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for batch in test_bar:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            # --- OPTIMISATION 2 : AMP (Autocast) pour l'inférence ---
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask)
                
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics
    # Map back encoded labels to original text labels
    label_map = df[['label', 'label_encoded']].drop_duplicates().sort_values('label_encoded')
    target_names = label_map['label'].tolist()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 5. Confusion Matrix Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8)) # Taille légèrement réduite pour un dataset binaire
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as confusion_matrix.png")
    # plt.show() # Sur un serveur distant comme Onyxia, plt.show() peut parfois bloquer, on commente par précaution.

if __name__ == "__main__":
    test()