import torch
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

# 1. Paramètres simulés (basés sur ta config)
EPOCHS = 10
STEPS_PER_EPOCH = 150  # Remplace par la taille réelle de ton train_loader si tu la connais, sinon 150 est très bien pour le graphique
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS = int(0.1 * TOTAL_STEPS) # 10% d'échauffement
PEAK_LR = 1e-4 # Le taux d'apprentissage de ta tête de classification

# 2. Création d'un "faux" modèle et optimiseur juste pour le test
dummy_model = torch.nn.Linear(1, 1)
optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=PEAK_LR)

# 3. Initialisation de ton scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TOTAL_STEPS
)

# 4. Simulation de l'entraînement et enregistrement des valeurs
lrs = []
for step in range(TOTAL_STEPS):
    # On enregistre le learning rate actuel
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    
    # On avance d'un pas (sans faire de backward pass !)
    optimizer.step()
    scheduler.step()

# 5. Création du graphique (Design propre et académique pour Beamer)
plt.figure(figsize=(8, 4))
plt.plot(lrs, color='#1f77b4', linewidth=2.5)

# Ajout de lignes visuelles pour expliquer les phases
plt.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.7, label='Fin du Warmup (10%)')
plt.fill_between(range(WARMUP_STEPS), 0, lrs[:WARMUP_STEPS], color='red', alpha=0.1)
plt.fill_between(range(WARMUP_STEPS, TOTAL_STEPS), 0, lrs[WARMUP_STEPS:], color='blue', alpha=0.05)

# Décoration du graphique
plt.title("Évolution du Taux d'Apprentissage (Cosine Annealing Warmup)", fontsize=14, fontweight='bold')
plt.xlabel("Étapes d'entraînement (Batches)", fontsize=12)
plt.ylabel("Taux d'Apprentissage (Learning Rate)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="upper right")
plt.tight_layout()

# Sauvegarde en PDF (Le format PDF est vectoriel, il ne pixelisera JAMAIS sur un Beamer)
plt.savefig('cosine_warmup.pdf', format='pdf', dpi=300)
print("Graphique sauvegardé avec succès sous le nom 'cosine_warmup.pdf'")