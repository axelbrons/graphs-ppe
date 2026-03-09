# src/train.py
import pandas as pd
import torch
import os
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src import config
from src.dataset import SolidityDataset
from src.model import GraphCodeBERTClassifier

def train():
    # 1. Load Data
    print(f"Loading data from {config.DATA_PATH}...")
    df = pd.read_parquet(config.DATA_PATH)
    num_labels = df['label_encoded'].nunique()
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=config.SEED, stratify=df['label_encoded'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=config.SEED, stratify=temp_df['label_encoded'])

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_dataset = SolidityDataset(train_df, tokenizer, config.MAX_SEQ_LEN, config.MAX_VAR_LEN)
    val_dataset = SolidityDataset(val_df, tokenizer, config.MAX_SEQ_LEN, config.MAX_VAR_LEN)
    
    #train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,        # Utilise 4 cœurs CPU en parallèle
        pin_memory=True,      # Prépare la mémoire pour un transfert ultra-rapide vers le GPU
        prefetch_factor=2     # Prépare 2 batchs d'avance
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    # 2. Initialize Model
    model = GraphCodeBERTClassifier(config.MODEL_NAME, num_labels)
    model.to(config.DEVICE)


    optimizer_grouped_parameters = [
        {'params': model.encoder.parameters(), 'lr': 1e-5}, # Corps lent
        {'params': model.classifier.parameters(), 'lr': 1e-4} # Tête rapide
    ]
    # optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    optimizer = AdamW(optimizer_grouped_parameters)

    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps), # 10% de montée
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()
    # weights_path = './data/class_weights.json'
    # if os.path.exists(weights_path):
    #     with open(weights_path, 'r') as f:
    #         data = json.load(f)
    #         class_weights = torch.tensor(data['weights']).to(config.DEVICE)
    #         print("Utilisation des poids de classes chargés depuis le fichier.")
    #         criterion = nn.CrossEntropyLoss(weight=class_weights)
    # else:
    #     print("Aucun fichier de poids trouvé, utilisation d'une Loss classique.")
    #     criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    scaler = torch.amp.GradScaler('cuda')

    # 3. Training Loop
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE) # (batch, 1, seq_len, seq_len)
            labels = batch['labels'].to(config.DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            # logits = model(input_ids, attention_mask)
            # loss = criterion(logits, labels)
            
            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            
            scale_before = scaler.get_scale()
            scaler.update()
            scale_after = scaler.get_scale()
            
            # Si le scale n'a pas baissé, l'optimizer a bien travaillé, on avance le scheduler
            if scale_before <= scale_after:
                scheduler.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 4. Validation Loop
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(config.DEVICE)
                attention_mask = batch['attention_mask'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)

                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                # logits = model(input_ids, attention_mask)
                # loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"Best model saved to {config.SAVE_PATH}")

    print("Training Complete.")


if __name__ == "__main__":
    train()