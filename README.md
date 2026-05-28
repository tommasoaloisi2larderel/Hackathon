# Projet Enedis - Analyse de courbes de charge residentielles

Ce projet applique plusieurs algorithmes d'IA a des courbes de charge residentielles Enedis `RES2-6-9 kVA`. L'objectif est d'identifier des profils de residences secondaires (RS) et residences principales (RP), de construire un classifieur a partir de ces labels, de prevoir une courbe J+1 et de generer des courbes synthetiques conditionnees par le type de consommateur.

## Donnees

Le fichier source utilise est :

```text
data/RES2-6-9.csv
```

Les valeurs sont des puissances en W au pas de 30 minutes. Elles sont converties en kW dans les notebooks. Pour calculer une energie, on multiplie par `0.5` heure.

Le dataset traite contient :

- 500 clients
- 8 736 000 lignes
- 17 472 pas de temps par client
- periode : 2023-10-31 23:00:00 a 2024-10-29 22:30:00

## Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── RES2-6-9.csv
│   ├── df_clean.parquet
│   ├── pivot.parquet
│   ├── features.parquet
│   ├── labels.csv
│   ├── synth_rs.npy
│   └── synth_rp.npy
├── models/
│   ├── classifier.pkl
│   ├── feature_cols.json
│   ├── forecast_meta.json
│   ├── kmeans.pkl
│   ├── lstm_forecaster.pt
│   ├── scaler_clf.pkl
│   ├── scaler_features.pkl
│   ├── scaler_ts.pkl
│   ├── scaler_vae.pkl
│   └── vae_conditional.pt
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_forecasting.ipynb
│   └── 05_generation.ipynb
├── README.md
└── requirements.txt
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app/streamlit_app.py
```

L'application Streamlit contient trois pages :

- Classification RS/RP d'un client
- Prevision J+1 avec baseline et LSTM
- Generation de courbes synthetiques RS/RP avec VAE conditionnel

## Methodologie

### 1. Exploration

Le notebook `01_exploration.ipynb` charge le CSV, convertit les puissances en kW, normalise les dates et construit une matrice `clients x timestamps`.

Les artefacts produits sont :

- `data/df_clean.parquet`
- `data/pivot.parquet`

### 2. Clustering

Le notebook `02_clustering.ipynb` construit des variables explicatives agregees par client, puis applique K-means.

Les principales features sont :

- consommation annuelle totale
- consommation moyenne journaliere
- variance et pic maximum
- ratio nuit/jour
- ratio ete/hiver
- part ete et part hiver
- nombre et taux de jours actifs
- nombre de jours quasi nuls
- consommation semaine/weekend
- ratio weekend/semaine

Le meilleur `k` selon silhouette est `2`, ce qui est coherent avec l'objectif metier RS/RP.

Labels obtenus :

- RP : 285 clients
- RS : 215 clients

Interpretation :

- Le cluster RS a une part ete plus elevee, moins de jours actifs et plus de jours quasi nuls.
- Le cluster RP a une consommation plus forte et plus reguliere, notamment en hiver.
- Le ratio weekend/semaine est affiche comme metrique descriptive, mais il discrimine peu les deux classes.

### 3. Classification

Le notebook `03_classification.ipynb` entraine plusieurs modeles supervises a partir des labels issus du clustering :

- Regression logistique
- Random Forest
- MLP

Le dataset de test est equilibre entre RS et RP. Le scaling est fait sans fuite de donnees : le scaler est ajuste uniquement sur le train.

Meilleur modele actuel :

- Random Forest
- Accuracy test : 1.000

Cette performance doit etre interpretee prudemment : le modele apprend a reproduire les labels crees par le clustering, pas une verite terrain externe.

### 4. Prevision

Le notebook `04_forecasting.ipynb` predit la courbe du lendemain a partir de 7 jours d'historique pour un client RP representatif.

Modeles compares :

- Baseline lineaire
- LSTM PyTorch

Resultats actuels :

- Baseline : MAE = 0.2964 kW, RMSE = 0.4299 kW
- LSTM : MAE = 0.1908 kW, RMSE = 0.3045 kW

Le LSTM ameliore la baseline d'environ 35.6% en MAE. La limite principale est que l'evaluation porte sur un seul client representatif.

### 5. Generation

Le notebook `05_generation.ipynb` entraine un VAE conditionnel sur le label RS/RP afin de generer des courbes journalieres de 48 points.

L'entrainement est equilibre entre classes. L'evaluation compare les courbes generees a un echantillon reel aleatoire de la meme classe.

Resultats actuels :

- RS : energie moyenne synthetique 13.20 kWh vs reel 12.86 kWh
- RP : energie moyenne synthetique 18.90 kWh vs reel 19.11 kWh
- KS energie : coherent sur les deux classes
- KS valeurs : encore mauvais, donc les courbes sont plus coherentes en energie journaliere qu'en distribution point par point

## Limites a mentionner

- Les labels RS/RP viennent du clustering, pas d'un label Enedis officiel.
- La classification mesure surtout la capacite a reproduire le clustering.
- La prevision est validee sur un seul client representatif.
- Le VAE genere des energies journalieres plausibles, mais la distribution fine des valeurs reste perfectible.

## Reexecuter les notebooks

Les notebooks doivent etre executes dans cet ordre :

```text
01_exploration.ipynb
02_clustering.ipynb
03_classification.ipynb
04_forecasting.ipynb
05_generation.ipynb
```

Chaque notebook sauvegarde les artefacts necessaires au suivant et au dashboard Streamlit.
