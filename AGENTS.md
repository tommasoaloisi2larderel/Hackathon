## Imported Claude Cowork project instructions

Voici le plan complet de A à Z.

---

## Phase 0 — Setup (1h)

**0.1 Environnement**
```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
```

**0.2 Installe les librairies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch plotly streamlit jupyter
pip freeze > requirements.txt
```

**0.3 Structure du repo Git**
```
projet-enedis/
├── data/
│   └── RES2-6-9kVA.csv
├── notebooks/          # brouillons exploratoires
│   ├── 01_exploration.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_forecasting.ipynb
│   └── 05_generation.ipynb
├── src/                # code propre réutilisé par Streamlit
│   ├── data_loader.py
│   ├── features.py
│   ├── clustering.py
│   ├── classification.py
│   ├── forecasting.py
│   └── generation.py
├── app/
│   └── streamlit_app.py
├── models/             # modèles sérialisés (.pkl, .pt)
├── README.md
└── requirements.txt
```

```bash
git init
git add .
git commit -m "init: project structure"
```

---

## Phase 1 — Exploration des données (2-3h)

**Notebook : `01_exploration.ipynb`**

**1.1 Charge et inspecte le CSV**
```python
import pandas as pd
df = pd.read_csv("data/RES2-6-9kVA.csv", sep=";")  # vérifie le séparateur
df.head()
df.info()
df.describe()
```

**1.2 Comprends la structure**
- Identifie les colonnes : probablement un ID client, une date/heure, une valeur de puissance (kW)
- Vérifie l'unité : puissance en kW à pas 30 min → **multiplier par 0.5 pour avoir des kWh**
- Vérifie les valeurs manquantes : `df.isnull().sum()`

**1.3 Reshape en matrice clients × timestamps**
```python
# Pivot : une ligne = un client, une colonne = un créneau horaire
df_pivot = df.pivot_table(index='id_client', columns='horodate', values='puissance_kw')
# tu auras ~17520 colonnes (48 pas/jour × 365 jours)
```

**1.4 Visualise quelques profils**
```python
import matplotlib.pyplot as plt
for i in range(5):
    plt.plot(df_pivot.iloc[i], alpha=0.5)
plt.title("Exemples de courbes de charge")
plt.show()
```
→ Tu dois voir des profils très différents : certains plats avec des pics ponctuels (RS), d'autres avec une consommation régulière (RP).

**1.5 Visualise la consommation moyenne par heure de la journée**
```python
# Moyenne sur tous les créneaux de la même heure
# (utile pour voir les patterns journaliers)
```

---

## Phase 2 — Feature Engineering (2h)

**Notebook : `02_clustering.ipynb`** (première partie)

K-means sur les 17520 colonnes brutes est possible mais bruité. Il vaut mieux construire des **features résumées** :

**2.1 Features à calculer par client**

| Feature | Description |
|---|---|
| `conso_totale` | Somme × 0.5 → kWh total |
| `conso_moyenne_jour` | kWh/jour moyen |
| `ratio_nuit_jour` | conso 22h-6h / conso 6h-22h |
| `variance` | variance de la puissance |
| `nb_jours_actifs` | jours avec conso > seuil |
| `pic_max` | puissance max observée |
| `conso_ete` | kWh sur juin-août |
| `conso_hiver` | kWh sur déc-fév |
| `ratio_ete_hiver` | **clé pour RS** : les RS consomment plus en été |

```python
features = pd.DataFrame({
    'conso_totale': df_pivot.sum(axis=1) * 0.5,
    'variance': df_pivot.var(axis=1),
    'ratio_nuit_jour': ...,
    'ratio_ete_hiver': ...,
    # etc.
})
```

**2.2 Normalise les features**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(features)
```

---

## Phase 3 — Clustering K-means (2h)

**Notebook : `02_clustering.ipynb`** (suite)

**3.1 Choisis k avec la méthode du coude**
```python
from sklearn.cluster import KMeans
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
plt.plot(range(2, 10), inertias, 'o-')
plt.xlabel('k')
plt.ylabel('Inertie')
```
→ Tu devrais voir un coude à k=2 (RS vs RP)

**3.2 Entraîne avec k=2**
```python
km = KMeans(n_clusters=2, random_state=42)
labels = km.fit_predict(X)
features['label'] = labels  # 0 ou 1
```

**3.3 Interprète les clusters**
```python
features.groupby('label').mean()
```
→ Identifie quel cluster est RS (ratio_ete_hiver élevé, nb_jours_actifs faible) et quel est RP.

**3.4 Visualise en 2D avec PCA**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='bwr', alpha=0.5)
```

**3.5 Sauvegarde les labels**
```python
features[['label']].to_csv("data/labels.csv")
```

```bash
git commit -m "feat: clustering kmeans RS/RP + features engineering"
```

---

## Phase 4 — Classification (3h)

**Notebook : `03_classification.ipynb`**

**4.1 Prépare le dataset équilibré**
```python
from sklearn.utils import resample
# Équilibre RS et RP (même nombre dans chaque classe)
rs = features[features['label']==0]
rp = features[features['label']==1]
n = min(len(rs), len(rp))
df_balanced = pd.concat([rs.sample(n), rp.sample(n)])
X_clf = df_balanced.drop('label', axis=1)
y_clf = df_balanced['label']
```

**4.2 Split train/test**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
```

**4.3 Entraîne 3 modèles**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: {score:.3f}")
```

**4.4 Matrice de confusion + métriques**
```python
from sklearn.metrics import classification_report, confusion_matrix
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
```

**4.5 Sauvegarde le meilleur modèle**
```python
import joblib
joblib.dump(best_model, "models/classifier.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```

```bash
git commit -m "feat: classification logistic/RF/MLP + evaluation"
```

---

## Phase 5 — Prévision (3-4h)

**Notebook : `04_forecasting.ipynb`**

**5.1 Choisis un client representatif** (un RP et un RS)

**5.2 Formate la série temporelle**
```python
# Une ligne = un jour, les colonnes = 48 créneaux de 30 min
# Ou : série 1D par client
client_series = df[df['id_client'] == client_id].sort_values('horodate')['puissance_kw']
```

**5.3 Modèle baseline : régression linéaire**
- Features : consommation des J-1, J-2, J-7 (même jour de la semaine)
- Target : consommation de J (48 valeurs)

**5.4 Modèle LSTM (PyTorch)**
```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=48, hidden_size=64, output_size=48):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```
- Input : 7 jours d'historique (fenêtre glissante)
- Output : le jour suivant (48 valeurs)
- Entraîne avec MSE loss

**5.5 Évalue**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f} kW")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f} kW")
```

**5.6 Visualise prédiction vs réel**

```bash
git commit -m "feat: forecasting LSTM J+1 + baseline lineaire"
```

---

## Phase 6 — Génération (3-4h)

**Notebook : `05_generation.ipynb`**

**6.1 Approche : VAE (Variational Autoencoder)**
- Input : courbe d'un jour (48 valeurs)
- Encode dans un espace latent conditionné au label RS/RP
- Décode pour générer de nouvelles courbes

```python
class VAE(nn.Module):
    def __init__(self, input_dim=48, latent_dim=8, label_dim=2):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim + label_dim, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim + label_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)
    
    def encode(self, x, c):
        h = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z, c):
        h = torch.relu(self.fc3(torch.cat([z, c], dim=1)))
        return torch.sigmoid(self.fc4(h))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
```

**6.2 Loss = reconstruction + KL divergence**

**6.3 Génère et vérifie la cohérence**
```python
# Génère 10 courbes RS synthétiques
label_rs = torch.tensor([[1, 0]] * 10, dtype=torch.float)
z = torch.randn(10, latent_dim)
synthetic = vae.decode(z, label_rs).detach().numpy()

# Compare avec les vraies courbes RS
plt.plot(synthetic.T, alpha=0.3, color='red', label='synthétique')
plt.plot(real_rs.T, alpha=0.3, color='blue', label='réel')
```

**6.4 Métriques de similarité**
- Distribution des valeurs (histogramme)
- Corrélation temporelle
- Énergie journalière moyenne

```bash
git commit -m "feat: generation VAE conditionnel RS/RP"
```

---

## Phase 7 — Dashboard Streamlit (3-4h)

**`app/streamlit_app.py`**

**7.1 Structure de l'app (3 pages)**
```python
import streamlit as st

st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Classification", "Prévision", "Génération"])

if page == "Classification":
    # Upload une courbe → prédit RS ou RP
    # Affiche la matrice de confusion, les métriques
    
elif page == "Prévision":
    # Sélectionne un client + une date
    # Affiche la prévision J+1 vs réel
    
elif page == "Génération":
    # Sélectionne RS ou RP
    # Génère N courbes synthétiques et les affiche
```

**7.2 Page Classification**
- Sélecteur de client dans le dataset
- Affiche sa courbe
- Prédit son type avec le modèle sauvegardé
- Affiche les métriques globales (accuracy, F1, confusion matrix avec Plotly)

**7.3 Page Prévision**
- Sélecteur de client + date de début
- Affiche l'historique (7 derniers jours)
- Affiche la prévision J+1 vs la réalité
- Affiche MAE/RMSE

**7.4 Page Génération**
- Bouton RS ou RP
- Slider pour le nombre de courbes
- Affiche les courbes synthétiques vs courbes réelles moyennes

```bash
streamlit run app/streamlit_app.py
git commit -m "feat: streamlit dashboard 3 pages"
```

---

## Phase 8 — Finalisation (1h)

**8.1 README.md**
```markdown
# Projet Enedis — Analyse de courbes de charge résidentielles

## Contexte
...

## Structure du projet
...

## Installation
pip install -r requirements.txt

## Lancer l'application
streamlit run app/streamlit_app.py

## Méthodologie
- Clustering : K-means sur features agrégées (ratio été/hiver, variance...)
- Classification : Random Forest (accuracy: XX%)
- Prévision : LSTM fenêtre 7 jours → J+1 (RMSE: XX kW)
- Génération : VAE conditionnel RS/RP
```

**8.2 Dernier commit**
```bash
git add .
git commit -m "docs: README + nettoyage final"
```

---

## Récap timeline

| Phase | Durée estimée |
|---|---|
| 0 — Setup | 1h |
| 1 — Exploration | 2-3h |
| 2-3 — Features + Clustering | 3-4h |
| 4 — Classification | 3h |
| 5 — Prévision | 3-4h |
| 6 — Génération | 3-4h |
| 7 — Streamlit | 3-4h |
| 8 — Finalisation | 1h |
| **Total** | **~20h** |

---

On attaque par quelle phase ? Je peux t'écrire le code complet de chaque étape au fur et à mesure.
