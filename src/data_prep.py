# src/data_prep.py
import pandas as pd
import os
import random
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from src import config

def create_optimized_dataset():
    print(f"Chargement du dataset original depuis {config.INPUT_CSV}...")
    try:
        df = pd.read_csv(config.INPUT_CSV)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {config.INPUT_CSV} est introuvable.")
        return

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # --- NOUVEAU : Suppression des classes indésirables ---
    labels_to_remove = [
        './Dataset/dangerous delegatecall (DE)/', 
        './Dataset/ether frozen (EF)'
    ]
    df = df[~df['label'].isin(labels_to_remove)]
    print(f"Classes supprimées : {labels_to_remove}")

    optimized_dfs = []
    unique_labels = df['label'].unique()

    print("\n--- ÉCHANTILLONNAGE DES VULNÉRABILITÉS ---")
    for label in unique_labels:
        subset = df[df['label'] == label]
        count = len(subset)
        
        # Règle 1 : Plafonner les grosses classes (Downsampling)
        if count > config.MAX_VULN_SAMPLES:
            sampled_subset = subset.sample(n=config.MAX_VULN_SAMPLES, random_state=config.SEED)
            print(f"{label}: Plafonné à {config.MAX_VULN_SAMPLES} (au lieu de {count})")
        else:
            # Règle 2 : Garder les petites classes telles quelles
            sampled_subset = subset
            print(f"{label}: Conservé tel quel ({count} exemples)")
            
        optimized_dfs.append(sampled_subset)

    print("\n--- TRAITEMENT DES CONTRATS SAINS (SAFE) ---")
    valid_files = [f for f in os.listdir(config.VALID_CONTRACTS_DIR) if f.endswith('.sol')]
    
    # Règle 3 : Plafonner le nombre de contrats sains
    if len(valid_files) > config.MAX_SAFE_SAMPLES:
        random.seed(config.SEED)
        selected_valid_files = random.sample(valid_files, config.MAX_SAFE_SAMPLES)
        print(f"Safe: Plafonné à {config.MAX_SAFE_SAMPLES} (sur {len(valid_files)} disponibles)")
    else:
        selected_valid_files = valid_files
        print(f"Safe: {len(valid_files)} contrats sains utilisés")
    
    valid_data = []
    for filename in selected_valid_files:
        file_path = os.path.join(config.VALID_CONTRACTS_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
            valid_data.append({
                'filename': filename,
                'code': code_content,
                'label': 'Safe' 
                # On ne met plus de label_encoded ici, on le fera globalement à la fin
            })
        except Exception as e:
            pass # On ignore les fichiers illisibles

    if valid_data:
        df_valid = pd.DataFrame(valid_data)
        optimized_dfs.append(df_valid)

    # 4. Fusionner tout le monde
    final_df = pd.concat(optimized_dfs, ignore_index=True)
    
    # --- NOUVEAU : Ré-encodage continu et propre (0 à 6) ---
    print("\n--- RÉ-ENCODAGE DES LABELS (0 à 6) ---")
    le = LabelEncoder()
    final_df['label_encoded'] = le.fit_transform(final_df['label'])
    
    # Mélanger
    final_df = final_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    
    # On s'assure de ne garder que les bonnes colonnes
    final_df = final_df[['filename', 'code', 'label', 'label_encoded']]

    # 5. Sauvegarde
    final_df.to_csv(config.DATA_PATH, index=False)
    
    print("\n" + "="*50)
    print(f"Succès ! Nouveau dataset créé : {config.DATA_PATH}")
    print(f"Nombre total de lignes : {len(final_df)}")
    
    # --- CALCUL DES POIDS POUR PYTORCH ---
    print("\n--- CALCUL DES POIDS DE CLASSES (CLASS WEIGHTS) ---")
    class_counts = final_df['label_encoded'].value_counts().sort_index()
    n_samples = len(final_df)
    n_classes = len(class_counts)
    
    weights_array = []
    print("Poids calculés par label :")
    for encoded_label, count in class_counts.items():
        # Retrouver le nom du label grâce au LabelEncoder
        label_name = le.inverse_transform([encoded_label])[0]
        weight = n_samples / (n_classes * count)
        weights_array.append(round(weight, 4))
        print(f"  - [{encoded_label}] {label_name} : {round(weight, 4)} ({count} samples)")
        
    # Mapping propre généré par le LabelEncoder
    label_map = {int(i): label for i, label in enumerate(le.classes_)}
    
    weights_data = {
        "weights": weights_array,
        "label_map": label_map
    }
    
    with open('./data/class_weights.json', 'w') as f:
        json.dump(weights_data, f)
    print("Poids et mapping sauvegardés dans ./data/class_weights.json")

if __name__ == "__main__":
    create_optimized_dataset()