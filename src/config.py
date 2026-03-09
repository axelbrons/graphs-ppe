# src/config.py
import os
import torch
import random
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- CHEMINS (PATHS) ---
INPUT_CSV = './data/SC_Vuln_8label.csv'
VALID_CONTRACTS_DIR = './data/valid'
DATA_PATH = './data/dataset_2l_v2.parquet'
SAVE_PATH = "best_model_v6.pt"

# --- HYPERPARAMÈTRES ---
SEED = 42
MODEL_NAME = "microsoft/graphcodebert-base"
MAX_SEQ_LEN = 512
MAX_VAR_LEN = 128
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
SAMPLES_PER_CLASS = 200
MAX_VULN_SAMPLES = 600
MAX_SAFE_SAMPLES = 3000

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
SAFE_REPO_NAME = 'thec00n/etherscan_verified_contracts'
SAFE_TARGET_COUNT = 3000

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    """Fixe l'aléatoire pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)