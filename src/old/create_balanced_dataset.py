import pandas as pd
import os
import random

# --- CONFIGURATION ---
INPUT_CSV = './data/SC_Vuln_8label.csv' # REMPLACE PAR LE NOM DE TON CSV ACTUEL
VALID_CONTRACTS_DIR = './data/valid'
OUTPUT_CSV = './data/dataset_9label_200_v1.csv'
SAMPLES_PER_CLASS = 200

def create_balanced_dataset():
    # 1. Charger le dataset existant (les vulnérabilités)
    print("Chargement du dataset original...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {INPUT_CSV} est introuvable.")
        return

    # On s'assure d'avoir uniquement les colonnes utiles
    # Note : On ignore 'Unnamed: 0' s'il existe
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # 2. Échantillonner 200 lignes pour chaque label existant
    print("Échantillonnage des classes existantes...")
    balanced_dfs = []
    
    # On récupère la liste des labels uniques
    unique_labels = df['label'].unique()
    
    for label in unique_labels:
        # On filtre le dataframe pour ce label spécifique
        subset = df[df['label'] == label]
        
        # Si on a assez de données, on en prend 200 au hasard
        if len(subset) >= SAMPLES_PER_CLASS:
            sampled_subset = subset.sample(n=SAMPLES_PER_CLASS, random_state=42)
        else:
            # Si on a moins de 200 lignes, on prend tout ce qu'il y a (ou on duplique, ici on prend tout)
            print(f"Attention : Le label '{label}' a moins de {SAMPLES_PER_CLASS} exemples ({len(subset)}).")
            sampled_subset = subset
            
        balanced_dfs.append(sampled_subset)

    # 3. Traiter les fichiers "Sains" (Valid)
    print("Traitement des fichiers sains...")
    valid_files = [f for f in os.listdir(VALID_CONTRACTS_DIR) if f.endswith('.sol')]
    
    if len(valid_files) < SAMPLES_PER_CLASS:
        print(f"Attention : Tu n'as que {len(valid_files)} fichiers valides dans {VALID_CONTRACTS_DIR}. Le script en utilisera autant que possible.")
        selected_valid_files = valid_files
    else:
        # On mélange et on prend 200 fichiers
        random.shuffle(valid_files)
        selected_valid_files = valid_files[:SAMPLES_PER_CLASS]

    # Déterminer le nouveau code encodé (max existant + 1)
    new_label_encoded = df['label_encoded'].max() + 1
    
    valid_data = []
    for filename in selected_valid_files:
        file_path = os.path.join(VALID_CONTRACTS_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
                
            valid_data.append({
                'filename': filename,
                'code': code_content,
                'label': 'Safe', # Nouveau Label Texte
                'label_encoded': new_label_encoded # Nouveau Label Numérique (ex: 9)
            })
        except Exception as e:
            print(f"Erreur de lecture {filename}: {e}")

    # Création du dataframe pour les sains
    df_valid = pd.DataFrame(valid_data)
    balanced_dfs.append(df_valid)

    # 4. Fusionner et Mélanger
    final_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Mélanger aléatoirement toutes les lignes (shuffle)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Garder uniquement les colonnes demandées
    final_df = final_df[['filename', 'code', 'label', 'label_encoded']]

    # 5. Sauvegarde
    final_df.to_csv(OUTPUT_CSV, index=False) # index=False pour ne pas recréer une colonne Unnamed
    
    print("-" * 30)
    print(f"Succès ! Dataset créé : {OUTPUT_CSV}")
    print(f"Nombre total de lignes : {len(final_df)}")
    print("\nRépartition des classes :")
    print(final_df['label'].value_counts())

# Lancer la fonction
if __name__ == "__main__":
    create_balanced_dataset()