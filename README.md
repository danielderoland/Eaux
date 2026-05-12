# Eaux
J'ai les débits d'eau d'une ville sur des points de sur des PR on appelle ça je me donne les consommations toutes les 5 min pendant 2 ans et j'aimerais faire un modèle prédictif en fonction du jour de de l'heure du jour du jour de la semaine du mois de l'année un avoir un modèle prédictif au au plus proche

## Modèle prédictif minimal

Le fichier `predictive_model.py` implémente un modèle léger basé uniquement sur les variables calendaires demandées :
- année
- mois
- jour du mois
- jour de la semaine
- heure/minute

Le modèle apprend des moyennes historiques par niveaux (du plus précis au plus général) puis applique un fallback automatique si une combinaison n'a jamais été vue.

### Format de données attendu (CSV)

Exemple de colonnes :
- `timestamp` (date/heure, format ISO ou `YYYY-MM-DD HH:MM[:SS]`)
- `consumption` (débit/consommation numérique)

### Exécution

```bash
python3 predictive_model.py --input data.csv --timestamp-col timestamp --value-col consumption --predict 2026-05-12T14:35:00
```

La commande affiche :
- le nombre d'observations chargées
- une MAE sur un split temporel (20% validation)
- la prédiction pour l'horodatage demandé
