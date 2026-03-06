# src/fetch_contracts.py
import os
from github import Github, Auth
from src import config

def fetch_safe_contracts():
    # 1. Vérification du Token
    if not config.GITHUB_TOKEN:
        print("Erreur : GITHUB_TOKEN introuvable. Vérifie ton fichier .env à la racine.")
        return

    # 2. Préparation du dossier cible
    os.makedirs(config.VALID_CONTRACTS_DIR, exist_ok=True)

    # 3. Authentification GitHub
    auth = Auth.Token(config.GITHUB_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(config.SAFE_REPO_NAME)

    print(f"Connexion au dépôt '{config.SAFE_REPO_NAME}' réussie. Démarrage du scan...")

    count = 0
    contents = repo.get_contents("")

    # 4. Boucle de téléchargement
    while contents and count < config.SAFE_TARGET_COUNT:
        file_content = contents.pop(0)
        
        # Si c'est un dossier, on ajoute son contenu à la file d'attente
        if file_content.type == "dir":
            print(f"... Exploration du dossier : {file_content.path}")
            try:
                contents.extend(repo.get_contents(file_content.path))
            except Exception as e:
                print(f"Erreur d'accès au dossier {file_content.path}: {e}")

        # Si c'est un fichier .sol, on le télécharge
        elif file_content.name.endswith(".sol"):
            
            try:
                code = file_content.decoded_content.decode('utf-8', errors='ignore')
                safe_name = file_content.name.replace("/", "_") 
                
                # Sauvegarde dans le dossier ./data/valid/
                file_path = os.path.join(config.VALID_CONTRACTS_DIR, safe_name)
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(code)
                
                count += 1
                print(f"[{count}/{config.SAFE_TARGET_COUNT}] SAUVEGARDÉ : {safe_name}")
                
            except Exception as e:
                print(f"Erreur lors du téléchargement de {file_content.name}: {e}")

    print("Fini ! Dataset de contrats sains téléchargé et prêt.")

if __name__ == "__main__":
    fetch_safe_contracts()