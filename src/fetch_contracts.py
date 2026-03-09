# src/fetch_contracts.py
import os
import requests # Pour des téléchargements ultra-rapides
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

    print(f"Connexion au dépôt '{config.SAFE_REPO_NAME}' réussie. Récupération de l'arborescence...")

    # --- LA MAGIE EST ICI ---
    # On récupère tout l'arbre du dépôt (bypasse la limite de 1000 fichiers)
    branch = repo.default_branch
    tree = repo.get_git_tree(branch, recursive=True).tree

    # On filtre pour ne garder que la liste des fichiers .sol
    sol_files = [item for item in tree if item.path.endswith('.sol') and item.type == 'blob']
    print(f"Trouvé {len(sol_files)} fichiers .sol au total sur le dépôt GitHub !")

    count = 0
    # 4. Boucle de téléchargement optimisée
    for item in sol_files:
        if count >= config.SAFE_TARGET_COUNT:
            break
            
        try:
            # On génère l'URL brute (comme quand tu cliques sur "Raw" sur GitHub)
            raw_url = f"https://raw.githubusercontent.com/{config.SAFE_REPO_NAME}/{branch}/{item.path}"
            
            # Téléchargement direct sans impacter le quota d'API GitHub
            response = requests.get(raw_url, timeout=10)
            
            if response.status_code == 200:
                code = response.text
                safe_name = item.path.replace("/", "_") 
                
                # Sauvegarde
                file_path = os.path.join(config.VALID_CONTRACTS_DIR, safe_name)
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(code)
                
                count += 1
                print(f"[{count}/{config.SAFE_TARGET_COUNT}] SAUVEGARDÉ : {safe_name}")
            else:
                print(f"Erreur HTTP {response.status_code} pour {item.path}")
                
        except Exception as e:
            print(f"Erreur lors du téléchargement de {item.path}: {e}")

    print("Fini ! Dataset de contrats sains téléchargé et prêt.")

if __name__ == "__main__":
    fetch_safe_contracts()