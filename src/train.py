import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from tree_sitter import Language, Parser
import tree_sitter_solidity as tssoldity

# --- CONFIGURATION & SEED ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/graphcodebert-base"
MAX_SEQ_LEN = 512
MAX_VAR_LEN = 128 # Max number of variable nodes
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
SAVE_PATH = "best_model.pt"
DATA_PATH = "./data/dataset_9label_200_v1.csv"

# --- TREE-SITTER SETUP ---
SOL_LANGUAGE = Language(tssoldity.language())
parser = Parser(SOL_LANGUAGE)

def get_variable_nodes(node, variables):
    # Find all identifier nodes (potential variables)
    if node.type == 'identifier' and node.child_count == 0:
        variables.append(node)
    for child in node.children:
        get_variable_nodes(child, variables)
    return variables

def extract_data_flow_edges(node, edges):
    # Assignment: x = y
    if node.type == 'assignment_expression':
        left_node = node.child_by_field_name('left')
        right_node = node.child_by_field_name('right')
        add_edges(left_node, right_node, edges)
    # Declaration: uint x = a
    elif node.type == 'variable_declaration_statement':
        declarations = []
        get_variable_nodes(node, declarations)
        if len(declarations) >= 2:
            target = declarations[0]
            sources = declarations[1:]
            for s in sources:
                edges.append((s.id, target.id))
    for child in node.children:
        extract_data_flow_edges(child, edges)

def add_edges(lhs, rhs, edges):
    if lhs and rhs:
        lhs_vars = get_variable_nodes(lhs, [])
        rhs_vars = get_variable_nodes(rhs, [])
        for source in rhs_vars:
            for target in lhs_vars:
                edges.append((source.id, target.id))

def build_graph_mask(seq_len, code_indices, var_indices, flow_edges, alignment_edges):
    # Initialiser avec une valeur très négative (interdit)
    mask = torch.full((seq_len, seq_len), -1e9)

    # --- RÈGLE 1 : CLS, SEP et Code ---
    # In GraphCodeBERT, CLS, SEP and Code tokens can all see each other.
    # We define special indices (CLS is at 0, first SEP is after code, last SEP is at the end)
    special_indices = [0, code_indices[-1] + 1, seq_len - 1]
    allowed_text = special_indices + code_indices
    
    for i in allowed_text:
        for j in allowed_text:
            mask[i, j] = 0

    # --- RÈGLE 2 : Data Flow (E) ---
    # Variable nodes can see themselves and their sources.
    for i in var_indices:
        mask[i, i] = 0
        
    for src_idx, tgt_idx in flow_edges:
        # src_idx and tgt_idx are absolute indices in the sequence
        if tgt_idx < seq_len and src_idx < seq_len:
            mask[tgt_idx, src_idx] = 0 

    # --- RÈGLE 3 : Alignement (E') ---
    # Bidirectional link between graph node and code token
    for var_idx, code_idx in alignment_edges:
        if var_idx < seq_len and code_idx < seq_len:
            mask[var_idx, code_idx] = 0
            mask[code_idx, var_idx] = 0

    return mask

class SolidityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_len=512, max_var_len=128):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_var_len = max_var_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = row['code']
        label = row['label_encoded']

        # 1. Parse AST and extract variables/flow
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node
        all_vars = get_variable_nodes(root, [])
        flow_edges_ast = []
        extract_data_flow_edges(root, flow_edges_ast)

        # 2. Tokenize code
        # We need to track token spans to align with AST nodes
        encoding = self.tokenizer(
            code,
            max_length=self.max_seq_len - self.max_var_len - 3, # CLS, 2 SEPs
            truncation=True,
            return_offsets_mapping=True,
            padding=False
        )
        
        code_ids = encoding['input_ids'] # Already has [CLS] and [SEP]
        offsets = encoding['offset_mapping']
        
        # Identify code tokens (excluding CLS and SEP)
        # Assuming CLS is at index 0 and SEP is at the end
        code_indices = list(range(1, len(code_ids) - 1))
        
        # 3. Handle Variable Nodes (V)
        # Limit the number of variable nodes to keep sequence length under control
        vars_to_use = all_vars[:self.max_var_len]
        var_names = [v.text.decode('utf8', errors='ignore') for v in vars_to_use]
        var_ids = self.tokenizer.convert_tokens_to_ids(var_names)
        
        # Total sequence: [CLS] + code + [SEP] + vars + [SEP]
        input_ids = code_ids + var_ids + [self.tokenizer.sep_token_id]
        
        # Sequence indices for variables
        var_start_idx = len(code_ids)
        var_indices = list(range(var_start_idx, var_start_idx + len(var_ids)))
        
        # 4. Map AST nodes to Sequence Indices
        node_id_to_seq_idx = {}
        alignment_edges = []
        
        # For each variable node in V, we find its corresponding token in C
        for i, var_node in enumerate(vars_to_use):
            var_seq_idx = var_indices[i]
            node_id_to_seq_idx[var_node.id] = var_seq_idx
            
            # Find which code token corresponds to this AST node
            # We look for the first token that overlaps with the node's range
            start_byte = var_node.start_byte
            for token_idx in code_indices:
                t_start, t_end = offsets[token_idx]
                if t_start <= start_byte < t_end:
                    alignment_edges.append((var_seq_idx, token_idx))
                    break

        # 5. Build Flow Edges for the sequence
        flow_edges = []
        for src_id, tgt_id in flow_edges_ast:
            if src_id in node_id_to_seq_idx and tgt_id in node_id_to_seq_idx:
                flow_edges.append((node_id_to_seq_idx[src_id], node_id_to_seq_idx[tgt_id]))

        # 6. Build Mask
        seq_len = len(input_ids)
        mask = build_graph_mask(seq_len, code_indices, var_indices, flow_edges, alignment_edges)

        # Padding
        padding_len = self.max_seq_len - seq_len
        if padding_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            # Extend mask: padded tokens are fully masked
            new_mask = torch.full((self.max_seq_len, self.max_seq_len), -1e9)
            new_mask[:seq_len, :seq_len] = mask
            mask = new_mask
        else:
            input_ids = input_ids[:self.max_seq_len]
            mask = mask[:self.max_seq_len, :self.max_seq_len]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': mask.unsqueeze(0), # (1, seq_len, seq_len)
            'labels': torch.tensor(label, dtype=torch.long)
        }

class GraphCodeBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels, freeze_backbone=False):
        super(GraphCodeBERTClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        # attention_mask here is the 4D mask (batch, 1, seq_len, seq_len)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    num_labels = df['label_encoded'].nunique()
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['label_encoded'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['label_encoded'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = SolidityDataset(train_df, tokenizer, MAX_SEQ_LEN, MAX_VAR_LEN)
    val_dataset = SolidityDataset(val_df, tokenizer, MAX_SEQ_LEN, MAX_VAR_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 2. Initialize Model
    model = GraphCodeBERTClassifier(MODEL_NAME, num_labels)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE) # (batch, 1, seq_len, seq_len)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 4. Validation Loop
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Best model saved to {SAVE_PATH}")

    print("Training Complete.")

if __name__ == "__main__":
    train()
