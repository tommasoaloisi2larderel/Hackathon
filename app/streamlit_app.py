import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import json
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

# ── Model definitions (must match training) ──────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=48, hidden_size=64, num_layers=2, output_size=48):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=48, label_dim=2, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),       nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),             nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=1))


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    pivot = pd.read_parquet(DATA / "pivot.parquet")
    features = pd.read_parquet(DATA / "features.parquet")
    df = pd.read_parquet(DATA / "df_clean.parquet")
    return pivot, features, df


@st.cache_resource
def load_classifier():
    clf = joblib.load(MODELS / "classifier.pkl")
    scaler = joblib.load(MODELS / "scaler_clf.pkl")
    with open(MODELS / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return clf, scaler, feature_cols


@st.cache_resource
def load_lstm():
    with open(MODELS / "forecast_meta.json") as f:
        meta = json.load(f)
    model = LSTMForecaster()
    model.load_state_dict(torch.load(MODELS / "lstm_forecaster.pt", map_location="cpu"))
    model.eval()
    scaler = joblib.load(MODELS / "scaler_ts.pkl")
    return model, scaler, meta


@st.cache_resource
def load_vae():
    vae = ConditionalVAE()
    vae.load_state_dict(torch.load(MODELS / "vae_conditional.pt", map_location="cpu"))
    vae.eval()
    scaler = joblib.load(MODELS / "scaler_vae.pkl")
    return vae, scaler


# ── Feature computation (mirrors notebook 02) ────────────────────────────────
def compute_features(serie: pd.Series) -> pd.DataFrame:
    """Compute aggregated features for a single client's time series."""
    cols = pd.to_datetime(serie.index)
    n_slots = 48
    n_days = len(serie) // n_slots

    mask_nuit = (cols.hour >= 22) | (cols.hour < 6)
    mask_ete = cols.month.isin([6, 7, 8])
    mask_hiver = cols.month.isin([12, 1, 2])

    cn = serie[mask_nuit].sum() * 0.5
    cj = serie[~mask_nuit].sum() * 0.5
    ce = serie[mask_ete].sum() * 0.5
    ch = serie[mask_hiver].sum() * 0.5
    conso_totale = serie.sum() * 0.5

    daily_kw = serie.iloc[:n_days * n_slots].to_numpy().reshape(n_days, n_slots)
    daily_kwh = daily_kw.sum(axis=1) * 0.5
    day_dates = pd.to_datetime(serie.index[:n_days * n_slots][::n_slots])
    is_weekend = day_dates.weekday >= 5
    is_weekday = ~is_weekend

    conso_weekend = daily_kwh[is_weekend].sum()
    conso_semaine = daily_kwh[is_weekday].sum()
    conso_moy_weekend = daily_kwh[is_weekend].mean()
    conso_moy_semaine = daily_kwh[is_weekday].mean()
    nb_jours_actifs = int((daily_kwh > 1.0).sum())

    return pd.DataFrame([{
        "conso_totale": conso_totale,
        "conso_moy_jour": conso_totale / n_days,
        "conso_moy_jour_actif": conso_totale / max(nb_jours_actifs, 1),
        "variance": serie.var(),
        "pic_max": serie.max(),
        "ratio_nuit_jour": cn / (cj + 1e-6),
        "ratio_ete_hiver": (ce + 1.0) / (ch + 1.0),
        "log_ratio_ete_hiver": np.log1p(ce) - np.log1p(ch),
        "part_ete": ce / (conso_totale + 1e-6),
        "part_hiver": ch / (conso_totale + 1e-6),
        "conso_ete": ce,
        "conso_hiver": ch,
        "conso_semaine": conso_semaine,
        "conso_weekend": conso_weekend,
        "conso_moy_semaine": conso_moy_semaine,
        "conso_moy_weekend": conso_moy_weekend,
        "ratio_weekend_semaine": conso_moy_weekend / (conso_moy_semaine + 1e-6),
        "ecart_weekend_semaine": conso_moy_weekend - conso_moy_semaine,
        "nb_jours_actifs": nb_jours_actifs,
        "taux_jours_actifs": nb_jours_actifs / n_days,
        "nb_jours_quasi_nuls": int((daily_kwh <= 0.5).sum()),
        "coeff_variation": serie.std() / (serie.mean() + 1e-6),
    }])


# ── App layout ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Enedis — Courbes de charge", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["🔍 Classification", "📈 Prévision", "✨ Génération"])

pivot, features, df = load_data()
all_ids = pivot.index.tolist()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔍 Classification":
    st.title("Classification RS / RP")
    st.markdown("Prédit si un client est une **Résidence Secondaire (RS)** ou une **Résidence Principale (RP)**.")

    clf, scaler_clf, feature_cols = load_classifier()

    col1, col2 = st.columns([1, 2])

    with col1:
        client_id = st.selectbox("Sélectionne un client", all_ids)
        true_label = features.loc[client_id, "label"] if client_id in features.index else "?"

        serie = pivot.loc[client_id]
        feat  = compute_features(serie)
        X     = scaler_clf.transform(feat[feature_cols])
        pred  = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]

        pred_label = "RS" if pred == 1 else "RP"
        color = "#e74c3c" if pred_label == "RS" else "#2980b9"

        st.markdown("---")
        st.metric("Label clustering (référence)", true_label)
        st.markdown(f"### Prédiction : <span style='color:{color}; font-size:2em'>{pred_label}</span>", unsafe_allow_html=True)
        st.progress(float(proba[pred]), text=f"Confiance : {proba[pred]:.1%}")

        st.markdown("---")
        st.markdown("**Features calculées**")
        st.dataframe(feat[feature_cols].T.rename(columns={0: "valeur"}).round(3), height=380)

    with col2:
        st.markdown("#### Courbe annuelle")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(serie.values, linewidth=0.4, color=color)
        ax.set_xlabel("Créneau (30 min)"); ax.set_ylabel("kW")
        ax.set_title(f"Client {client_id} — {pred_label}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Profil journalier moyen")
        hour_vals = df[df["ID"] == client_id].copy()
        hour_vals["hour"] = hour_vals["horodate"].dt.hour + hour_vals["horodate"].dt.minute / 60
        hourly = hour_vals.groupby("hour")["kw"].mean()
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(hourly.index, hourly.values, color=color, linewidth=2)
        ax2.fill_between(hourly.index, hourly.values, alpha=0.15, color=color)
        ax2.set_xlabel("Heure"); ax2.set_ylabel("kW moyen")
        ax2.set_xticks(range(0, 25, 2)); ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close()

    # Métriques globales
    with st.expander("📊 Performances globales du classifieur (sur tous les clients labelisés)"):
        X_all = scaler_clf.transform(features[feature_cols])
        y_true = (features["label"] == "RS").astype(int).values
        y_pred_all = clf.predict(X_all)
        acc = (y_true == y_pred_all).mean()
        st.metric("Accuracy globale", f"{acc:.1%}")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        cm = confusion_matrix(y_true, y_pred_all)
        ConfusionMatrixDisplay(cm, display_labels=["RP", "RS"]).plot(ax=ax3, colorbar=False)
        st.pyplot(fig3)
        plt.close()
        st.text(classification_report(y_true, y_pred_all, target_names=["RP", "RS"]))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉVISION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Prévision":
    st.title("Prévision J+1 (LSTM)")
    st.markdown("Prédit la courbe de consommation du lendemain à partir de **7 jours d'historique**.")

    lstm, scaler_ts, meta = load_lstm()
    trained_client = int(meta["client_id"])

    N_SLOTS = 48
    WINDOW  = 7

    # Prépare la matrice jours × 48 pour le client entraîné
    serie = pivot.loc[trained_client].values
    n_days = len(serie) // N_SLOTS
    matrix = serie[:n_days * N_SLOTS].reshape(n_days, N_SLOTS)
    matrix_norm = scaler_ts.transform(matrix)

    # Sélecteur de jour (dans la plage de test : 80% → fin)
    split = int(n_days * 0.8)
    max_day = n_days - WINDOW - 1

    col1, col2 = st.columns([1, 3])
    with col1:
        st.info(f"Client entraîné : **{trained_client}**\n\n(label : {features.loc[trained_client, 'label']})")
        day_idx = st.slider("Jour à prévoir (index)", min_value=split, max_value=max_day, value=split + 5)
        st.metric("MAE LSTM",     f"{meta['mae']:.4f} kW")
        st.metric("MAE Baseline", f"{meta['mae_baseline']:.4f} kW")
        st.metric("Gain vs baseline", f"{(1 - meta['mae']/meta['mae_baseline'])*100:.1f}%")

    with col2:
        window_norm = matrix_norm[day_idx - WINDOW: day_idx]  # (7, 48)
        target_norm = matrix_norm[day_idx]                     # (48,)

        # LSTM prediction
        x_tensor = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)  # (1, 7, 48)
        with torch.no_grad():
            pred_norm = lstm(x_tensor).numpy()[0]

        # Baseline J-1
        bl_norm = matrix_norm[day_idx - 1]

        # Dénormalise
        real_kw = scaler_ts.inverse_transform(target_norm.reshape(1, -1))[0]
        pred_kw = scaler_ts.inverse_transform(pred_norm.reshape(1, -1))[0]
        bl_kw   = scaler_ts.inverse_transform(bl_norm.reshape(1, -1))[0]

        mae_lstm = mean_absolute_error(real_kw, pred_kw)
        mae_bl   = mean_absolute_error(real_kw, bl_kw)

        t = np.arange(N_SLOTS) * 0.5

        # Historique 7 jours
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        hist_kw = scaler_ts.inverse_transform(window_norm)
        for i, day in enumerate(hist_kw):
            axes[0].plot(t, day, alpha=0.4, linewidth=0.8, color="steelblue")
        axes[0].plot(t, real_kw, color="black", linewidth=2, label="Jour J (réel)")
        axes[0].set_title("Historique 7 jours + jour cible")
        axes[0].set_xlabel("Heure"); axes[0].set_ylabel("kW")
        axes[0].set_xticks(range(0, 25, 2))
        axes[0].legend()

        axes[1].plot(t, real_kw, color="black",    linewidth=2,   label=f"Réel")
        axes[1].plot(t, bl_kw,   color="coral",    linewidth=1.5, linestyle="--", label=f"Baseline J-1 (MAE={mae_bl:.3f})")
        axes[1].plot(t, pred_kw, color="steelblue",linewidth=1.5, linestyle="--", label=f"LSTM (MAE={mae_lstm:.3f})")
        axes[1].set_title("Prévision J+1")
        axes[1].set_xlabel("Heure"); axes[1].set_ylabel("kW")
        axes[1].set_xticks(range(0, 25, 2))
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Courbe de performances sur toute la période de test
    with st.expander("📊 Performances sur toute la période de test"):
        maes_lstm, maes_bl = [], []
        for d in range(split + WINDOW, n_days):
            w  = matrix_norm[d - WINDOW: d]
            tg = matrix_norm[d]
            xt = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pn = lstm(xt).numpy()[0]
            rk = scaler_ts.inverse_transform(tg.reshape(1, -1))[0]
            pk = scaler_ts.inverse_transform(pn.reshape(1, -1))[0]
            bk = scaler_ts.inverse_transform(matrix_norm[d-1].reshape(1, -1))[0]
            maes_lstm.append(mean_absolute_error(rk, pk))
            maes_bl.append(mean_absolute_error(rk, bk))

        fig4, ax4 = plt.subplots(figsize=(12, 3))
        ax4.plot(maes_bl,   alpha=0.6, color="coral",     label=f"Baseline (moy={np.mean(maes_bl):.3f} kW)")
        ax4.plot(maes_lstm, alpha=0.6, color="steelblue", label=f"LSTM (moy={np.mean(maes_lstm):.3f} kW)")
        ax4.set_xlabel("Jour de test"); ax4.set_ylabel("MAE (kW)")
        ax4.set_title("MAE jour par jour sur la période de test")
        ax4.legend()
        st.pyplot(fig4)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — GÉNÉRATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "✨ Génération":
    st.title("Génération de courbes synthétiques (VAE)")
    st.markdown("Génère des courbes de charge réalistes conditionnées sur le type RS ou RP.")

    vae, scaler_vae = load_vae()

    col1, col2 = st.columns([1, 3])

    with col1:
        label_choice = st.radio("Type à générer", ["RS", "RP"])
        n_gen = st.slider("Nombre de courbes", 5, 100, 20)
        n_real = st.slider("Courbes réelles à afficher", 5, 50, 15)
        seed = st.number_input("Seed aléatoire", value=42, step=1)
        generate_btn = st.button("Générer ✨", use_container_width=True)

    with col2:
        if generate_btn or True:  # affiche par défaut
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

            # Génère
            c = torch.zeros(n_gen, 2)
            c[:, 0 if label_choice == "RS" else 1] = 1
            z = torch.randn(n_gen, vae.latent_dim)
            with torch.no_grad():
                synth_norm = vae.decode(z, c).numpy()
            synth_kw = scaler_vae.inverse_transform(synth_norm)

            # Vraies courbes
            ids_label = features[features["label"] == label_choice].index
            N_SLOTS = 48
            real_curves = []
            for cid in ids_label:
                s = pivot.loc[cid].values
                nd = len(s) // N_SLOTS
                days = s[:nd * N_SLOTS].reshape(nd, N_SLOTS)
                active = days[(days.sum(axis=1) * 0.5) > 1.0]
                if len(active) > 0:
                    real_curves.append(active)
            real_kw = np.vstack(real_curves)
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(len(real_kw), size=min(n_real, len(real_kw)), replace=False)
            real_sample = real_kw[idx]

            t = np.arange(N_SLOTS) * 0.5
            color_s = "#e74c3c" if label_choice == "RS" else "#2980b9"
            color_r = "#c0392b" if label_choice == "RS" else "#1a5276"

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Overlay
            for c_ in synth_kw:
                axes[0].plot(t, c_, alpha=0.2, linewidth=0.7, color=color_s)
            for c_ in real_sample:
                axes[0].plot(t, c_, alpha=0.2, linewidth=0.7, color="gray")
            axes[0].plot(t, synth_kw.mean(axis=0), color=color_s, linewidth=2.5, label="Synthétique moy.")
            axes[0].plot(t, real_sample.mean(axis=0), color="black", linewidth=2.5, label="Réel moy.")
            axes[0].set_title(f"Profils {label_choice} — synthétique vs réel")
            axes[0].set_xlabel("Heure"); axes[0].set_ylabel("kW")
            axes[0].set_xticks(range(0, 25, 2))
            axes[0].legend()

            # Distribution énergie journalière
            e_synth = synth_kw.sum(axis=1) * 0.5
            e_real  = real_sample.sum(axis=1) * 0.5
            axes[1].hist(e_real,  bins=25, alpha=0.6, color="gray",    density=True, label=f"Réel (moy={e_real.mean():.1f} kWh)")
            axes[1].hist(e_synth, bins=25, alpha=0.6, color=color_s, density=True, label=f"Synthétique (moy={e_synth.mean():.1f} kWh)")
            axes[1].set_title("Distribution énergie journalière (kWh)")
            axes[1].set_xlabel("kWh"); axes[1].set_ylabel("Densité")
            axes[1].legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Métriques
            from scipy.stats import ks_2samp
            idx_ks = rng.choice(len(real_kw), size=min(500, len(real_kw)), replace=False)
            ks_stat, ks_pval = ks_2samp(synth_kw.flatten(), real_kw[idx_ks].flatten())
            corr = np.corrcoef(synth_kw.mean(axis=0), real_sample.mean(axis=0))[0, 1]

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Énergie moy. synthétique", f"{e_synth.mean():.2f} kWh")
            mc2.metric("Corrélation profil moyen",  f"{corr:.4f}")
            mc3.metric("Test KS (p-value)",          f"{ks_pval:.4f}")
