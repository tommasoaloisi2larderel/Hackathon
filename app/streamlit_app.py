import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
MODELS = ROOT / "models"

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.grid"       : True,
    "grid.alpha"      : 0.25,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "font.size"       : 10,
})
COLOR_RS = "#e74c3c"
COLOR_RP = "#2980b9"
N_SLOTS  = 48
WINDOW   = 7


# ── Model definitions (must match notebooks) ──────────────────────────────────
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
    pivot    = pd.read_parquet(DATA / "pivot.parquet")
    features = pd.read_parquet(DATA / "features.parquet")
    df       = pd.read_parquet(DATA / "df_clean.parquet")
    return pivot, features, df


@st.cache_resource
def load_classifier():
    clf    = joblib.load(MODELS / "classifier.pkl")
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


# ── Feature engineering — must match what classifier was trained on ────────────
def compute_features(serie: pd.Series) -> pd.DataFrame:
    """Toutes les features calculées à l'entraînement (feature_cols.json est la source de vérité)."""
    cols       = pd.to_datetime(serie.index)
    n_slots    = 48
    n_days     = len(serie) // n_slots

    mask_nuit  = (cols.hour >= 22) | (cols.hour < 6)
    mask_ete   = cols.month.isin([6, 7, 8])
    mask_hiver = cols.month.isin([12, 1, 2])

    cn           = serie[mask_nuit].sum()  * 0.5
    cj           = serie[~mask_nuit].sum() * 0.5
    ce           = serie[mask_ete].sum()   * 0.5
    ch           = serie[mask_hiver].sum() * 0.5
    conso_totale = serie.sum() * 0.5

    daily_kw          = serie.iloc[:n_days * n_slots].to_numpy().reshape(n_days, n_slots)
    daily_kwh         = daily_kw.sum(axis=1) * 0.5
    day_dates         = pd.to_datetime(serie.index[:n_days * n_slots][::n_slots])
    is_weekend        = day_dates.weekday >= 5
    is_weekday        = ~is_weekend
    nb_jours_actifs   = int((daily_kwh > 1.0).sum())
    conso_moy_weekend = daily_kwh[is_weekend].mean() if is_weekend.any() else 0.0
    conso_moy_semaine = daily_kwh[is_weekday].mean() if is_weekday.any() else 0.0

    return pd.DataFrame([{
        "conso_totale"          : conso_totale,
        "conso_moy_jour"        : conso_totale / n_days,
        "conso_moy_jour_actif"  : conso_totale / max(nb_jours_actifs, 1),
        "variance"              : serie.var(),
        "pic_max"               : serie.max(),
        "ratio_nuit_jour"       : cn / (cj + 1e-6),
        "ratio_ete_hiver"       : (ce + 1.0) / (ch + 1.0),
        "log_ratio_ete_hiver"   : np.log1p(ce) - np.log1p(ch),
        "part_ete"              : ce / (conso_totale + 1e-6),
        "part_hiver"            : ch / (conso_totale + 1e-6),
        "conso_ete"             : ce,
        "conso_hiver"           : ch,
        "conso_semaine"         : daily_kwh[is_weekday].sum(),
        "conso_weekend"         : daily_kwh[is_weekend].sum(),
        "conso_moy_semaine"     : conso_moy_semaine,
        "conso_moy_weekend"     : conso_moy_weekend,
        "ratio_weekend_semaine" : conso_moy_weekend / (conso_moy_semaine + 1e-6),
        "ecart_weekend_semaine" : conso_moy_weekend - conso_moy_semaine,
        "nb_jours_actifs"       : nb_jours_actifs,
        "taux_jours_actifs"     : nb_jours_actifs / n_days,
        "nb_jours_quasi_nuls"   : int((daily_kwh <= 0.5).sum()),
        "coeff_variation"       : serie.std() / (serie.mean() + 1e-6),
    }])


# ── Precompute MAE series for the forecasting page (cached) ───────────────────
@st.cache_data
def compute_test_maes(trained_client: int):
    """MAE LSTM et baseline sur toute la période de test."""
    _, scaler_ts, meta = load_lstm()
    lstm, _, _         = load_lstm()
    pivot, _, _        = load_data()

    serie       = pivot.loc[trained_client].values
    n_days      = len(serie) // N_SLOTS
    matrix      = serie[:n_days * N_SLOTS].reshape(n_days, N_SLOTS)
    matrix_norm = scaler_ts.transform(matrix)
    split       = int(n_days * 0.8)

    maes_lstm, maes_bl = [], []
    for d in range(split + WINDOW, n_days):
        w  = matrix_norm[d - WINDOW: d]
        tg = matrix_norm[d]
        xt = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pn = lstm(xt).numpy()[0]
        rk = scaler_ts.inverse_transform(tg.reshape(1, -1))[0]
        pk = scaler_ts.inverse_transform(pn.reshape(1, -1))[0]
        bk = scaler_ts.inverse_transform(matrix_norm[d - 1].reshape(1, -1))[0]
        maes_lstm.append(mean_absolute_error(rk, pk))
        maes_bl.append(mean_absolute_error(rk, bk))

    return np.array(maes_lstm), np.array(maes_bl)


# ── App ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Enedis — Courbes de charge", layout="wide")

st.sidebar.title("Enedis · Analyse")
st.sidebar.caption("Courbes de charge résidentielles")
page = st.sidebar.radio(
    "Page",
    ["🔍 Classification", "📈 Prévision", "✨ Génération"],
    label_visibility="collapsed",
)

pivot, features, df = load_data()
all_ids = pivot.index.tolist()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔍 Classification":
    st.title("Classification — Résidence Secondaire vs Principale")

    clf, scaler_clf, feature_cols = load_classifier()

    # ── Sélection client & prédiction ─────────────────────────────────────────
    client_id  = st.selectbox("Client", all_ids, format_func=lambda x: f"ID {x}")
    serie      = pivot.loc[client_id]
    feat       = compute_features(serie)
    X          = scaler_clf.transform(feat[feature_cols])
    pred       = clf.predict(X)[0]
    proba      = clf.predict_proba(X)[0]
    pred_label = "RS" if pred == 1 else "RP"
    confidence = float(proba[pred])
    true_label = features.loc[client_id, "label"] if client_id in features.index else "?"
    color      = COLOR_RS if pred_label == "RS" else COLOR_RP

    # ── Résultat ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Prédiction", pred_label)
    c2.metric("Confiance",  f"{confidence:.1%}")
    c3.metric("Label de référence", true_label)

    if true_label != "?":
        if pred_label == true_label:
            st.success("Accord avec le clustering")
        else:
            st.warning("Désaccord avec le clustering")

    st.divider()

    # ── Visualisations ────────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(serie.values, linewidth=0.35, color=color, alpha=0.8)
        ax.set_ylabel("kW"); ax.set_xlabel("Créneau (30 min)")
        ax.set_title(f"Courbe annuelle — client {client_id}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with right:
        client_df = df[df["ID"] == client_id].copy()
        client_df["hour"] = (
            client_df["horodate"].dt.hour + client_df["horodate"].dt.minute / 60
        )
        hourly = client_df.groupby("hour")["kw"].mean()
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.plot(hourly.index, hourly.values, color=color, linewidth=2)
        ax2.fill_between(hourly.index, hourly.values, alpha=0.12, color=color)
        ax2.set_ylabel("kW moyen"); ax2.set_xlabel("Heure")
        ax2.set_title("Profil journalier moyen")
        ax2.set_xticks(range(0, 25, 2))
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Performances globales ─────────────────────────────────────────────────
    with st.expander("Performances globales du classifieur"):
        X_all      = scaler_clf.transform(features[feature_cols])
        y_true     = (features["label"] == "RS").astype(int).values
        y_pred_all = clf.predict(X_all)
        acc        = (y_true == y_pred_all).mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc:.1%}")
        c2.metric("Clients RS", int(y_true.sum()))
        c3.metric("Clients RP", int((1 - y_true).sum()))

        col_cm, col_report = st.columns([1, 2])
        with col_cm:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            ConfusionMatrixDisplay(
                confusion_matrix(y_true, y_pred_all),
                display_labels=["RP", "RS"]
            ).plot(ax=ax_cm, colorbar=False)
            ax_cm.set_title("Matrice de confusion")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()
        with col_report:
            st.code(classification_report(y_true, y_pred_all, target_names=["RP", "RS"]))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉVISION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Prévision":
    st.title("Prévision J+1 — LSTM")

    lstm, scaler_ts, meta = load_lstm()
    trained_client        = int(meta["client_id"])
    client_label          = features.loc[trained_client, "label"]

    serie       = pivot.loc[trained_client].values
    n_days      = len(serie) // N_SLOTS
    matrix      = serie[:n_days * N_SLOTS].reshape(n_days, N_SLOTS)
    matrix_norm = scaler_ts.transform(matrix)
    split       = int(n_days * 0.8)

    left, right = st.columns([1, 3], gap="large")

    with left:
        st.markdown(f"**Client entraîné :** {trained_client} ({client_label})")
        st.markdown(f"**Période de test :** jours {split} → {n_days - 1}")
        st.markdown("---")
        day_idx = st.slider(
            "Jour à prévoir",
            min_value=split + WINDOW,
            max_value=n_days - 1,
            value=split + WINDOW,
        )
        st.markdown("---")
        st.metric("MAE LSTM",     f"{meta['mae']:.3f} kW")
        st.metric("MAE Baseline (J−1)", f"{meta['mae_baseline']:.3f} kW",
                  delta=f"−{(1 - meta['mae'] / meta['mae_baseline']) * 100:.1f}%",
                  delta_color="inverse")

    with right:
        window_norm = matrix_norm[day_idx - WINDOW: day_idx]
        target_norm = matrix_norm[day_idx]

        x_tensor = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_norm = lstm(x_tensor).numpy()[0]

        bl_norm  = matrix_norm[day_idx - 1]
        real_kw  = scaler_ts.inverse_transform(target_norm.reshape(1, -1))[0]
        pred_kw  = scaler_ts.inverse_transform(pred_norm.reshape(1, -1))[0]
        bl_kw    = scaler_ts.inverse_transform(bl_norm.reshape(1, -1))[0]
        hist_kw  = scaler_ts.inverse_transform(window_norm)

        mae_this_lstm = mean_absolute_error(real_kw, pred_kw)
        mae_this_bl   = mean_absolute_error(real_kw, bl_kw)

        t = np.arange(N_SLOTS) * 0.5

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        for day in hist_kw:
            axes[0].plot(t, day, linewidth=0.7, color="steelblue", alpha=0.35)
        axes[0].plot(t, real_kw, color="black", linewidth=2, label="Jour J (réel)")
        axes[0].set_title(f"Historique J−{WINDOW} à J−1")
        axes[0].set_xlabel("Heure"); axes[0].set_ylabel("kW")
        axes[0].set_xticks(range(0, 25, 2))
        axes[0].legend()

        axes[1].plot(t, real_kw, color="black",      linewidth=2,   label="Réel")
        axes[1].plot(t, bl_kw,   color="coral",      linewidth=1.5, linestyle="--",
                     label=f"Baseline J−1 (MAE {mae_this_bl:.3f} kW)")
        axes[1].plot(t, pred_kw, color="steelblue",  linewidth=1.5, linestyle="--",
                     label=f"LSTM       (MAE {mae_this_lstm:.3f} kW)")
        axes[1].set_title(f"Prévision jour {day_idx}")
        axes[1].set_xlabel("Heure"); axes[1].set_ylabel("kW")
        axes[1].set_xticks(range(0, 25, 2))
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with st.expander("MAE sur toute la période de test"):
        maes_lstm, maes_bl = compute_test_maes(trained_client)

        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.plot(maes_bl,   color="coral",     alpha=0.7,
                 label=f"Baseline J−1  (moyenne {maes_bl.mean():.3f} kW)")
        ax2.plot(maes_lstm, color="steelblue", alpha=0.7,
                 label=f"LSTM          (moyenne {maes_lstm.mean():.3f} kW)")
        ax2.axhline(maes_lstm.mean(), color="steelblue", linewidth=0.8, linestyle=":")
        ax2.axhline(maes_bl.mean(),   color="coral",     linewidth=0.8, linestyle=":")
        ax2.set_xlabel("Jour de test"); ax2.set_ylabel("MAE (kW)")
        ax2.set_title("MAE journalière — période de test")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — GÉNÉRATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "✨ Génération":
    st.title("Génération de courbes synthétiques — VAE conditionnel")

    vae, scaler_vae = load_vae()

    left, right = st.columns([1, 3], gap="large")

    with left:
        label_choice = st.radio("Type", ["RS", "RP"])
        n_gen        = st.slider("Courbes à générer",         5, 100, 20)
        n_real       = st.slider("Courbes réelles à afficher", 5,  50, 15)
        seed         = st.number_input("Seed", value=42, step=1)

    # Génération
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    cond = torch.zeros(n_gen, 2)
    cond[:, 0 if label_choice == "RS" else 1] = 1
    z = torch.randn(n_gen, vae.latent_dim)
    with torch.no_grad():
        synth_norm = vae.decode(z, cond).numpy()
    synth_kw = scaler_vae.inverse_transform(synth_norm)

    # Vraies courbes du même type
    ids_label   = features[features["label"] == label_choice].index
    real_curves = []
    for cid in ids_label:
        s      = pivot.loc[cid].values
        n_days = len(s) // N_SLOTS
        days   = s[:n_days * N_SLOTS].reshape(n_days, N_SLOTS)
        active = days[(days.sum(axis=1) * 0.5) > 1.0]
        if len(active) > 0:
            real_curves.append(active)
    real_kw = np.vstack(real_curves)

    rng         = np.random.default_rng(int(seed))
    idx         = rng.choice(len(real_kw), size=min(n_real, len(real_kw)), replace=False)
    real_sample = real_kw[idx]

    t       = np.arange(N_SLOTS) * 0.5
    color_s = COLOR_RS if label_choice == "RS" else COLOR_RP

    with right:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overlay courbes
        for c_ in synth_kw:
            axes[0].plot(t, c_, linewidth=0.6, color=color_s, alpha=0.2)
        for c_ in real_sample:
            axes[0].plot(t, c_, linewidth=0.6, color="gray",  alpha=0.2)
        axes[0].plot(t, synth_kw.mean(axis=0),    color=color_s, linewidth=2.5, label="Synthétique (moy.)")
        axes[0].plot(t, real_sample.mean(axis=0), color="black", linewidth=2.5, label="Réel (moy.)")
        axes[0].set_title(f"Profils {label_choice} — synthétique vs réel")
        axes[0].set_xlabel("Heure"); axes[0].set_ylabel("kW")
        axes[0].set_xticks(range(0, 25, 2))
        axes[0].legend()

        # Distribution énergies journalières
        e_synth = synth_kw.sum(axis=1) * 0.5
        e_real  = real_sample.sum(axis=1) * 0.5
        axes[1].hist(e_real,  bins=25, alpha=0.6, color="gray",  density=True,
                     label=f"Réel        (moy. {e_real.mean():.1f} kWh)")
        axes[1].hist(e_synth, bins=25, alpha=0.6, color=color_s, density=True,
                     label=f"Synthétique (moy. {e_synth.mean():.1f} kWh)")
        axes[1].set_title("Distribution énergie journalière")
        axes[1].set_xlabel("kWh/jour"); axes[1].set_ylabel("Densité")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Métriques de similarité
        idx_ks = rng.choice(len(real_kw), size=min(500, len(real_kw)), replace=False)
        ks_stat, ks_pval = ks_2samp(synth_kw.flatten(), real_kw[idx_ks].flatten())
        corr = np.corrcoef(synth_kw.mean(axis=0), real_sample.mean(axis=0))[0, 1]

        st.markdown("**Métriques de similarité**")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Énergie moy. synthétique", f"{e_synth.mean():.2f} kWh",
                   delta=f"{e_synth.mean() - e_real.mean():+.2f} vs réel")
        mc2.metric("Corrélation profil moyen", f"{corr:.4f}")
        mc3.metric("Test KS — p-value",         f"{ks_pval:.4f}",
                   help="p > 0.05 : les distributions sont statistiquement similaires")
