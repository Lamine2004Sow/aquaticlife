# Projet M2 — Émergence de morphologies et stratégies de nage

## 1. Problématique
Comment des formes corporelles et des stratégies de propulsion aquatique émergent-elles lorsqu’on co-évolue simultanément la morphologie (segments, muscles) et le contrôle moteur (réseau neuronal) dans un fluide 2D visqueux, sans imposer de solution a priori ?

## 2. Hypothèses
- Des morphologies allongées et segmentées apparaîtront pour maximiser la portée propulsive.
- La fréquence/phase des oscillations musculaires s’alignera sur la fenêtre de Reynolds visqueux choisie.
- La co-évolution produira des compromis distincts vitesse / coût énergétique / stabilité.
- RL pur et neuro-évolution mèneront à des stratégies différentes (ex. RL → contrôleurs plus stables, NE → diversité morphologique accrue).

## 3. Mesures & objectifs
- **Performance primaire** : distance nette parcourue (m).
- **Coût** : énergie consommée (∑|force × vitesse| sur le temps).
- **Stabilité** : variance angulaire / sorties du centre de masse.
- **Diversity bonus** : distance morphologique (vecteur paramètres).
- **Score multi-objectif** : combinaison pondérée ou Pareto (NSGA-II).

## 4. Environnement physique (2D)
- Fluide visqueux, faible nombre de Reynolds → dynamique sur-amortie (Stokes) ou semi-inertielle simplifiée.
- Résistance fluide : force linéaire `F = -k_drag * v` par segment.
- Courants optionnels : champ vectoriel simple (sinusoïde spatiale).
- Limites : domaine rectangulaire, conditions périodiques ou murs doux (pénalités).

## 5. Morphologie
- Chaîne de `N` segments rigides 2D, masses `m_i`, longueurs `l_i`.
- Articulations à ressort/rotule avec raideur `k_joint` et amortissement `c_joint`.
- Muscles antagonistes par articulation, amplitude max `a_max`.
- Paramètres évolués : `N`, `l_i`, `m_i`, `k_joint`, `c_joint`, position/force des muscles.
- Gènes encodés dans un vecteur réel borné + masque booléen (présence muscle).

## 6. Contrôleur neuronal
- Entrées : angles relatifs, vitesses angulaires, vitesse COM, orientation globale, capteurs courants locaux, bruit.
- Sorties : activation musculaire par articulation (continu [-1, 1]).
- Architectures : MLP (2–3 couches) ou petit GRU.
- Apprentissage : 
  - RL (PPO ou SAC) avec récompense distance – pénalité énergie – pénalité instabilité.
  - Neuro-évolution (ES/GA) pour co-évoluer poids + morphologie.

## 7. Boucle d’optimisation
```
population ← init_random()
repeat for generations:
    fitness ← simulate(population)
    parents ← select(fitness)
    offspring ← mutate(parents, morpho+weights)
    if RL_hybrid: fine-tune offspring controllers with PPO
    population ← offspring
```
- Mutation morphologique : bruit gaussien borné, ajout/suppression segment (si N min/max respectés).
- Sélection : tournoi + élitisme, ou NSGA-II si multi-objectif explicite.

## 8. Visualisations clés
- Animation des individus champions par génération.
- Courbes distance/énergie/stabilité vs générations.
- Heatmaps efficacité (distance/énergie) par morphologie (N, longueur totale, rigidité).
- Diagramme de Pareto vitesse vs coût.
- Diversité morphologique (t-SNE/UMAP du vecteur de gènes).

## 9. Stack proposée
- Python 3.11, PyTorch ou JAX (CPU suffisant).
- NumPy/JAX pour la physique 2D; option PyTorch autograd pour forces.
- Hydra + OmegaConf pour la config d’expériences.
- Matplotlib/Plotly pour visualisations; FFmpeg pour vidéos.
- Tests avec pytest.

## 10. Plan de travail (commits prévus)
1. **Scaffolding** : package `aquaticlife/`, config Hydra, scripts d’expérience, README enrichi.  
2. **Physique 2D minimale** : segments, drag, joints, muscles, intégrateur semi-implicite.  
3. **Environnement RL** : wrapper Gymn-like, reward et normalisation obs.  
4. **Boucle évolution** : GA/ES multi-objectifs, mutations morpho+poids.  
5. **Hybride RL** : PPO fine-tuning, gestion des resets de morphologie.  
6. **Visualisation** : export trajets, rendu Matplotlib, vidéos.  
7. **Expériences** : scripts reproductibles + tableaux résultats.  
8. **Rapport** : synthèse des résultats et limites.

## 11. Risques & contournements
- Simulation fluide réaliste trop coûteuse → modèle drag simplifié ou Stokes analytique.
- Explosion des gradients avec morphologies changeantes → clip + normalisation paramètres.
- Dérive morphologique (taille infinie) → contraintes fortes sur N, longueurs, masses.
- Overfitting contrôle à une morphologie unique → diversité morphologique encouragée par novelty search.

## 12. Livrables finaux
- Code reproductible (scripts + configs).
- Vidéos d’évolution et trajectoires.
- Figures (heatmaps, Pareto, diversité).
- Rapport court (8–10 pages) décrivant méthode, résultats, limites.
