## Projet IMA207 — Self-Supervised Learning

Pour un entrainement sur Slurm, utiliser le fichier job_script.sh avec la commande sbatch job_script.sh

Pour notre implémentation nous avons travaillé sur plusieurs branche sur le gitlab, avec chacun une branche. 

Notre gitlab est organisé comme ci-dessous. Pour cette archive, nous avons tout regroupé en un seul dossier.

- **`multigpu_slurm_transfo`** contient la base de notre code implémentant **SimCLR**, avec un contrôle complet des hyperparamètres.  
  Elle inclut également la possibilité d'enregistrer et de reprendre l'entraînement à partir d'une sauvegarde.  
  Cette branche permet d’exécuter le modèle sur plusieurs GPU, compatible avec **SLURM** ainsi qu’avec les GPU de l’infrastructure de l’école grâce au script exécutable `job_script.sh`.  
  Les résultats sont envoyés vers **Weights & Biases (wandb)**, soit en direct si l’environnement est connecté à Internet, soit localement, avec une synchronisation différée via `sync_wandb.sh`.

- **`multi_gpu_slurm_an`** est la branche utilisée pour tester différentes têtes de projection dans notre modèle.

- **`barlow_twins`** ajoute la possibilité d’utiliser **Barlow Twins** : cette branche implémente la fonction de perte correspondante ainsi qu’une tête de projection adaptée.

Dans chacune de ces branches, les fichiers sont organisés de la manière suivante :

- **`utils/`**  
  Contient des fonctions utilitaires : calcul des métriques, boucle d’entraînement avecgestion des checkpoints, intégration avec wandb.

- **`data/`**  
  Contient les scripts de téléchargement, de préparation et de gestion des jeux de données utilisés dans le projet.

- **`models/`**  
  Implémente les architectures de réseaux de neurones utilisées (SimCLR, ResNet, variantes...).

- **`lars_optim/`**  
  Contient l’implémentation de l’optimiseur **LARS**.

- **`loss/`**  
  Contient la ou les fonctions de perte utilisées (**InfoNCE**, **Barlow Twins**).

Le notebook `DINOv2_CLIP_PathMNIST_Eval.ipynb` nous a permis de comparer nos résultats avec les modèles **DINOv2** et **CLIP**.

Enfin, pour synchroniser les fichiers wandb sur le cluster avec slurm, et le site : 
- Nous synchronisons les archives sur le git, ce qui permet d'y accéder en local sur son ordinateur.
- Il y a alors un script synchro_focus.sh qui permet de synchroniser une run au choix (facile pour créer des projets wandb distincts)


Auteurs : 
Maël LE GUILLOUZIC,
Arthur NUVOLONI,
Noa ANDRE,
Mohamed MOHAMED EL BECHIR

Encadrant : Pietro GORI