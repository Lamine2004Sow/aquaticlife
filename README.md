# Émergence de morphologies et stratégies de nage

Projet M2 – simulation co-évolution morphologie–contrôle dans un fluide 2D visqueux.

## Objectif
Faire émerger conjointement la forme du corps et la stratégie de nage d’organismes artificiels plongés dans un environnement aquatique. Aucun schéma de nage n’est codé en dur : la physique + l’apprentissage déterminent les solutions.

## Pitch rapide
- **Organisme minimal** : segments articulés + muscles contractiles.
- **Contrôleur neuronal** : petit MLP ou RNN entraîné par RL/neuro-évolution.
- **Morphologie paramétrique** : nombre de segments, longueurs, masses, rigidités, muscles.
- **Environnement** : fluide visqueux 2D avec résistance, courants simples et cibles.
- **Sélection** : distance parcourue, énergie consommée, stabilité, diversité morphologique.

## Pourquoi ça compte
Mettre en évidence comment des formes et des rythmes de nage émergent d’une pression de sélection simple (se déplacer efficacement), en comparant RL vs neuro-évolution et en visualisant la diversité des solutions.

## État des lieux
- Code en cours de rédaction.
- Voir `docs/projet.md` pour le plan détaillé et la feuille de route.

## Roadmap (résumé)
1) Spécification scientifique + métriques d’évaluation.
2) Squelette simulateur fluide 2D + organisme articulé.
3) Boucle d’apprentissage (RL + évolution morphologique).
4) Visualisations : trajectoires, morphologies, heatmaps d’efficacité.
5) Expériences et comparaisons RL vs neuro-évolution.

## Démarrage rapide (prototype GA)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/evolve.py
# PPO minimal (nécessite torch) :
pip install -e '.[rl]'
python scripts/train_rl.py
# Visualisation temps réel (pygame) :
pip install -e '.[viz]'
python scripts/viewer.py
```

## Arborescence
- `aquaticlife/physics/` : segments, joints, drag.
- `aquaticlife/envs/` : environnement gym-like.
- `aquaticlife/control/` : contrôleurs neuronaux (NumPy pour l’instant).
- `aquaticlife/evolution/` : GA minimal pour la morphologie.
- `aquaticlife/rl/` : agent PPO torch.
- `aquaticlife/visualization/` : viewer pygame temps réel.
- `scripts/` : exemples d’exécution.
- `configs/` : paramètres par défaut (Hydra-ready).
- `docs/` : spécification détaillée.
