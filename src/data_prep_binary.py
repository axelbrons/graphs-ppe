# src/data_prep_binary.py
import pandas as pd
import os
import random
import json
from sklearn.preprocessing import LabelEncoder
from src import config

def create_binary_dataset():
    print(f"Chargement du dataset original depuis {config.INPUT_CSV}...")
    try:
        df = pd.read_csv(config.INPUT_CSV)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {config.INPUT_CSV} est introuvable.")
        return

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # --- 1. Regroupement de toutes les vulnérabilités en une seule classe ---
    print("\n--- REGROUPEMENT DES VULNÉRABILITÉS ---")
    df['label'] = 'Vulnerable'
    
    total_vulnerable = len(df)
    print(f"Nombre total de contrats vulnérables trouvés : {total_vulnerable}")

    # --- 2. Traitement des contrats sains (Safe) ---
    print("\n--- TRAITEMENT DES CONTRATS SAINS (SAFE) ---")
    valid_files = [f for f in os.listdir(config.VALID_CONTRACTS_DIR) if f.endswith('.sol')]
    
    # Pour l'instant, on prend tous les fichiers Safe disponibles (on filtrera à 1000 après)
    valid_data = []
    for filename in valid_files:
        file_path = os.path.join(config.VALID_CONTRACTS_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
            valid_data.append({
                'filename': filename,
                'code': code_content,
                'label': 'Safe' 
            })
        except Exception as e:
            pass 

    df_valid = pd.DataFrame(valid_data)

    # --- 3. Fusionner et préparer le dataset final ---
    final_df = pd.concat([df, df_valid], ignore_index=True)
    
    # Ré-encodage binaire
    print("\n--- ENCODAGE BINAIRE ---")
    le = LabelEncoder()
    final_df['label_encoded'] = le.fit_transform(final_df['label'])
    
    # --- NOUVEAU : LIMITATION STRICTE À 1000 DE CHAQUE ---
    print("\n--- RÉDUCTION À 1000 EXEMPLES PAR CLASSE ---")
    df_vuln = final_df[final_df['label'] == 'Vulnerable']
    df_safe = final_df[final_df['label'] == 'Safe']

    # On prend 1000 exemples (ou le maximum disponible si on en a moins de 1000)
    n_vuln = min(1000, len(df_vuln))
    n_safe = min(1000, len(df_safe))

    df_vuln_sampled = df_vuln.sample(n=n_vuln, random_state=config.SEED)
    df_safe_sampled = df_safe.sample(n=n_safe, random_state=config.SEED)

    # On refusionne nos 2000 lignes
    final_df = pd.concat([df_vuln_sampled, df_safe_sampled], ignore_index=True)

    # --- 4. Mélanger le tout ---
    final_df = final_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    final_df = final_df[['filename', 'code', 'label', 'label_encoded']]

    # --- 5. Sauvegarde ---
    final_df.to_csv(config.DATA_PATH, index=False)
    
    print("\n" + "="*50)
    print(f"Succès ! Nouveau dataset Binaire Rapide créé : {config.DATA_PATH}")
    print(f"Nombre total de lignes : {len(final_df)}")
    print(f"Répartition exacte :\n{final_df['label'].value_counts()}")
    
    # --- 6. Calcul des poids ---
    class_counts = final_df['label_encoded'].value_counts().sort_index()
    n_samples = len(final_df)
    n_classes = len(class_counts)
    
    weights_array = []
    for encoded_label, count in class_counts.items():
        label_name = le.inverse_transform([encoded_label])[0]
        weight = n_samples / (n_classes * count)
        weights_array.append(round(weight, 4))
        
    label_map = {int(i): label for i, label in enumerate(le.classes_)}
    
    weights_data = {
        "weights": weights_array,
        "label_map": label_map
    }
    
    with open('./data/class_weights.json', 'w') as f:
        json.dump(weights_data, f)
    print("Mapping (0/1) sauvegardé dans ./data/class_weights.json")

if __name__ == "__main__":
    create_binary_dataset()