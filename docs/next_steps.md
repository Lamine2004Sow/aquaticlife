# Prochaines étapes de développement

## Technique
- Implémenter un entraîneur PPO (PyTorch) branché sur `rollout_episode` et compatible avec des morphologies variables (normalisation dynamique des obs).
- Ajouter un pas d'intégration plus réaliste (overdamped / Stokes explicite) et vérifier la stabilité numérique.
- Introduire un bruit de courant spatial (champ vectoriel) et des cibles à atteindre.
- Ajouter des rendus vidéo (Matplotlib/FFmpeg) pour les meilleurs individus à chaque génération.
- Écrire des tests rapides (pytest) pour la cinématique `forward_kinematics`, la stabilité de `step_dynamics` et la cohérence des récompenses.

## Expérimentations
- Comparer GA seul vs GA + fine-tuning PPO (hybride) sur 50 générations.
- Étudier l'impact du nombre de segments max et de la raideur des joints sur l'efficacité.
- Tracer les Pareto vitesse/énergie et la diversité morphologique (t-SNE du vecteur de gènes).
- Explorer l'ajout d'un bonus de nouveauté (novelty search) pour éviter la convergence prématurée.

## Hygiène projet
- Ajouter CI légère (lint + pytest) dès que les tests sont en place.
- Documenter les hyperparamètres clés dans `configs/` et fournir un script d sweep.
