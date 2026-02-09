import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from train import SolidityDataset, GraphCodeBERTClassifier, MODEL_NAME, MAX_SEQ_LEN, MAX_VAR_LEN, BATCH_SIZE, SEED, DEVICE, DATA_PATH, SAVE_PATH

def test():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # We must use the same split to get the test set
    _, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['label_encoded'])
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['label_encoded'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = SolidityDataset(test_df, tokenizer, MAX_SEQ_LEN, MAX_VAR_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 2. Load Model
    num_labels = df['label_encoded'].nunique()
    model = GraphCodeBERTClassifier(MODEL_NAME, num_labels)
    
    if not os.path.exists(SAVE_PATH):
        print(f"Error: Model file {SAVE_PATH} not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Evaluation
    all_preds = []
    all_labels = []
    test_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for batch in test_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics
    # Map back encoded labels to original text labels
    label_map = df[['label', 'label_encoded']].drop_duplicates().sort_values('label_encoded')
    target_names = label_map['label'].tolist()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 5. Confusion Matrix Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    import os
    test()
