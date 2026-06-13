import sys
import os

# ── Chemins absolus ────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
sys.path.append(os.path.join(BASE_DIR, 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

BLUE       = "#1F3864"
LIGHT_BLUE = "#2E75B6"
RED        = "#C0392B"
GREEN      = "#1E8449"
ORANGE     = "#D68910"
GRAY       = "#7F8C8D"

# ══════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES ET MODÈLES
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn_bin'] = (df['Churn'] == 'Yes').astype(int)
    return df

@st.cache_resource
def load_models():
    xgb_model    = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    cat_model    = joblib.load(os.path.join(MODELS_DIR, 'catboost_model.pkl'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    return xgb_model, cat_model, feature_cols

df = load_data()
xgb_model, cat_model, feature_cols = load_models()

# ══════════════════════════════════════════════════════════════
# SIDEBAR — FILTRES GLOBAUX
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🔧 Filtres")

contrat_filter = st.sidebar.multiselect(
    "Type de contrat",
    options=df['Contract'].unique().tolist(),
    default=df['Contract'].unique().tolist()
)

internet_filter = st.sidebar.multiselect(
    "Service Internet",
    options=df['InternetService'].unique().tolist(),
    default=df['InternetService'].unique().tolist()
)

tenure_range = st.sidebar.slider(
    "Durée d'abonnement (mois)",
    int(df['tenure'].min()),
    int(df['tenure'].max()),
    (0, 72)
)

model_choice = st.sidebar.selectbox("Modèle ML", ["XGBoost", "CatBoost"])
model = xgb_model if model_choice == "XGBoost" else cat_model

# Appliquer les filtres
df_filtered = df[
    df['Contract'].isin(contrat_filter) &
    df['InternetService'].isin(internet_filter) &
    df['tenure'].between(tenure_range[0], tenure_range[1])
].copy()

# ══════════════════════════════════════════════════════════════
# TITRE
# ══════════════════════════════════════════════════════════════
st.title("📊 Dashboard Analytique — Prédiction du Churn Client")
st.markdown(f"*Dataset filtré : **{len(df_filtered):,}** clients sur {len(df):,} total*")
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1 — KPIs
# ══════════════════════════════════════════════════════════════
st.header("📌 Indicateurs Clés (KPIs)")

total      = len(df_filtered)
churners   = df_filtered['Churn_bin'].sum()
non_churn  = total - churners
taux_churn = churners / total * 100 if total > 0 else 0
rev_risque = df_filtered[df_filtered['Churn'] == 'Yes']['MonthlyCharges'].sum()
rev_total  = df_filtered['MonthlyCharges'].sum()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("👥 Total Clients",    f"{total:,}")
with col2:
    st.metric("🔴 Clients Churners", f"{churners:,}",
              delta=f"{taux_churn:.1f}% du total", delta_color="inverse")
with col3:
    st.metric("🟢 Clients Fidèles",  f"{non_churn:,}")
with col4:
    st.metric("📉 Taux de Churn",    f"{taux_churn:.1f}%")
with col5:
    st.metric("💸 Revenu à Risque",  f"${rev_risque:,.0f}/mois",
              delta=f"{rev_risque/rev_total*100:.1f}% du CA" if rev_total > 0 else "N/A",
              delta_color="inverse")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — DISTRIBUTION & CHURN PAR SEGMENT
# ══════════════════════════════════════════════════════════════
st.header("📈 Analyse de la Distribution du Churn")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Répartition Churn / No Churn")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sizes  = [non_churn, churners]
    colors = [GREEN, RED]
    labels = [f"No Churn\n{non_churn:,} ({100-taux_churn:.1f}%)",
              f"Churn\n{churners:,} ({taux_churn:.1f}%)"]
    wedges, _ = ax1.pie(sizes, colors=colors, startangle=90,
                        wedgeprops=dict(width=0.55))
    ax1.legend(wedges, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)
    ax1.set_title("Distribution de la Variable Cible", fontsize=11, color=BLUE)
    st.pyplot(fig1)
    plt.close()

with col_b:
    st.subheader("Taux de Churn par Contrat")
    churn_contrat = (
        df_filtered.groupby('Contract')['Churn_bin']
        .agg(['mean', 'count']).reset_index()
    )
    churn_contrat['mean'] *= 100

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bars = ax2.bar(churn_contrat['Contract'], churn_contrat['mean'],
                   color=[RED if x > 30 else ORANGE if x > 15 else GREEN
                          for x in churn_contrat['mean']],
                   edgecolor='white', linewidth=1.2)
    for bar, val, n in zip(bars, churn_contrat['mean'], churn_contrat['count']):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{val:.1f}%\n(n={n:,})", ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel("Taux de Churn (%)")
    ax2.set_title("Churn selon le Type de Contrat", fontsize=11, color=BLUE)
    ax2.set_ylim(0, max(churn_contrat['mean']) * 1.3)
    ax2.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig2)
    plt.close()

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Churn par Service Internet")
    churn_net = (
        df_filtered.groupby('InternetService')['Churn_bin']
        .mean().reset_index()
    )
    churn_net['Churn_bin'] *= 100

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    colors_net = [RED if v > 30 else ORANGE if v > 15 else GREEN
                  for v in churn_net['Churn_bin']]
    ax3.barh(churn_net['InternetService'], churn_net['Churn_bin'],
             color=colors_net, edgecolor='white')
    for i, v in enumerate(churn_net['Churn_bin']):
        ax3.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=10)
    ax3.set_xlabel("Taux de Churn (%)")
    ax3.set_title("Churn par Type de Service Internet", fontsize=11, color=BLUE)
    ax3.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig3)
    plt.close()

with col_d:
    st.subheader("Churn selon la Durée d'Abonnement")
    df_filtered['tenure_group'] = pd.cut(
        df_filtered['tenure'],
        bins=[0, 6, 12, 24, 48, 72],
        labels=['0-6 mois', '7-12 mois', '13-24 mois', '25-48 mois', '49-72 mois']
    )
    churn_tenure = (
        df_filtered.groupby('tenure_group', observed=True)['Churn_bin']
        .mean() * 100
    ).reset_index()

    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.plot(churn_tenure['tenure_group'], churn_tenure['Churn_bin'],
             marker='o', color=LIGHT_BLUE, linewidth=2.5, markersize=8)
    ax4.fill_between(range(len(churn_tenure)), churn_tenure['Churn_bin'],
                     alpha=0.15, color=LIGHT_BLUE)
    ax4.set_xticks(range(len(churn_tenure)))
    ax4.set_xticklabels(churn_tenure['tenure_group'], rotation=20, ha='right')
    ax4.set_ylabel("Taux de Churn (%)")
    ax4.set_title("Evolution du Churn par Ancienneté", fontsize=11, color=BLUE)
    ax4.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate(churn_tenure['Churn_bin']):
        ax4.annotate(f"{v:.1f}%", (i, v), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=9)
    st.pyplot(fig4)
    plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — ANALYSE FINANCIÈRE
# ══════════════════════════════════════════════════════════════
st.header("💰 Analyse Financière")

# Prédictions sur les données filtrées
df_pred = df_filtered.copy()
for col in feature_cols:
    if col not in df_pred.columns:
        df_pred[col] = 0

probas = model.predict_proba(df_pred[feature_cols])[:, 1]
df_filtered = df_filtered.copy()
df_filtered['churn_proba'] = probas
df_filtered['risk_segment'] = pd.cut(
    df_filtered['churn_proba'],
    bins=[-0.01, 0.40, 0.70, 1.01],
    labels=['🟢 Faible (<40%)', '🟡 Moyen (40-70%)', '🔴 Élevé (>70%)']
)

col_e, col_f = st.columns(2)

with col_e:
    st.subheader("Distribution des Charges Mensuelles")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    churn_yes = df_filtered[df_filtered['Churn'] == 'Yes']['MonthlyCharges']
    churn_no  = df_filtered[df_filtered['Churn'] == 'No']['MonthlyCharges']
    ax5.hist(churn_no,  bins=30, alpha=0.6, color=GREEN, label='No Churn', density=True)
    ax5.hist(churn_yes, bins=30, alpha=0.6, color=RED,   label='Churn',    density=True)
    ax5.axvline(churn_yes.mean(), color=RED,   linestyle='--', linewidth=1.5,
                label=f'Moy. Churn: ${churn_yes.mean():.0f}')
    ax5.axvline(churn_no.mean(),  color=GREEN, linestyle='--', linewidth=1.5,
                label=f'Moy. No Churn: ${churn_no.mean():.0f}')
    ax5.set_xlabel("Monthly Charges ($)")
    ax5.set_ylabel("Densité")
    ax5.set_title("Charges Mensuelles : Churn vs No Churn", fontsize=11, color=BLUE)
    ax5.legend(fontsize=8)
    ax5.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig5)
    plt.close()

with col_f:
    st.subheader("Revenu Mensuel par Segment de Risque")
    rev_segment = df_filtered.groupby('risk_segment', observed=True)['MonthlyCharges'].agg(
        ['sum', 'count', 'mean']
    ).reset_index()

    fig6, ax6 = plt.subplots(figsize=(6, 4))
    colors_seg = [GREEN, ORANGE, RED]
    bars = ax6.bar(rev_segment['risk_segment'], rev_segment['sum'],
                   color=colors_seg, edgecolor='white', linewidth=1.2)
    for bar, row in zip(bars, rev_segment.itertuples()):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 200,
                 f"${row.sum:,.0f}\n({row.count} clients)",
                 ha='center', va='bottom', fontsize=8)
    ax6.set_ylabel("Revenu Mensuel Total ($)")
    ax6.set_title("CA Mensuel par Niveau de Risque", fontsize=11, color=BLUE)
    ax6.spines[['top', 'right']].set_visible(False)
    plt.xticks(fontsize=8)
    st.pyplot(fig6)
    plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE IMPORTANCE & CORRÉLATIONS
# ══════════════════════════════════════════════════════════════
st.header("🔍 Feature Importance & Corrélations")

col_g, col_h = st.columns(2)

with col_g:
    st.subheader(f"Top 15 Features — {model_choice}")
    importances = model.feature_importances_
    top_n   = 15
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_cols[i] for i in indices]
    top_vals     = importances[indices]

    fig7, ax7 = plt.subplots(figsize=(6, 5))
    colors_fi = [RED if v > np.percentile(top_vals, 75)
                 else LIGHT_BLUE if v > np.percentile(top_vals, 50)
                 else GRAY for v in top_vals]
    ax7.barh(top_features, top_vals, color=colors_fi, edgecolor='white')
    ax7.set_xlabel("Importance")
    ax7.set_title(f"Feature Importance — {model_choice}", fontsize=11, color=BLUE)
    ax7.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig7)
    plt.close()

with col_h:
    st.subheader("Matrice de Corrélation")
    num_vars = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_bin', 'SeniorCitizen']
    corr     = df_filtered[num_vars].corr()

    fig8, ax8 = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlBu_r',
                center=0, mask=mask, ax=ax8,
                linewidths=0.5, annot_kws={'size': 10})
    ax8.set_title("Corrélations entre Variables Numériques", fontsize=11, color=BLUE)
    st.pyplot(fig8)
    plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 5 — SEGMENTATION CLIENTS À RISQUE
# ══════════════════════════════════════════════════════════════
st.header("🎯 Segmentation des Clients à Risque")

seuil = st.slider("Seuil de probabilité de churn", 0.0, 1.0, 0.70, step=0.05)
clients_risque = df_filtered[df_filtered['churn_proba'] >= seuil].copy()

st.markdown(f"**{len(clients_risque)} clients** ont une probabilité de churn ≥ {seuil:.0%}")
st.markdown(f"Revenu mensuel en jeu : **${clients_risque['MonthlyCharges'].sum():,.0f}**")

col_i, col_j = st.columns(2)

with col_i:
    st.subheader("Répartition par Contrat (Clients à Risque)")
    if len(clients_risque) > 0:
        risque_contrat = clients_risque['Contract'].value_counts()
        fig9, ax9 = plt.subplots(figsize=(5, 4))
        ax9.pie(risque_contrat.values, labels=risque_contrat.index,
                autopct='%1.1f%%', colors=[RED, ORANGE, GREEN], startangle=90)
        ax9.set_title("Contrats des Clients à Risque", fontsize=11, color=BLUE)
        st.pyplot(fig9)
        plt.close()

with col_j:
    st.subheader("Top 10 Clients les Plus à Risque")
    if len(clients_risque) > 0:
        cols_display = ['gender', 'tenure', 'Contract', 'MonthlyCharges',
                        'InternetService', 'churn_proba']
        top10 = clients_risque.nlargest(10, 'churn_proba')[cols_display].copy()
        top10['churn_proba'] = (top10['churn_proba'] * 100).round(1).astype(str) + '%'
        top10.columns = ['Genre', 'Tenure', 'Contrat',
                         'Charges/mois ($)', 'Internet', 'Proba Churn']
        st.dataframe(top10, use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 6 — RECOMMANDATIONS AUTOMATISÉES
# ══════════════════════════════════════════════════════════════
st.header("💡 Recommandations Automatisées par Segment")

n_eleve  = len(df_filtered[df_filtered['churn_proba'] >= 0.70])
n_moyen  = len(df_filtered[(df_filtered['churn_proba'] >= 0.40) &
                            (df_filtered['churn_proba'] <  0.70)])
n_faible = len(df_filtered[df_filtered['churn_proba'] < 0.40])

col_k, col_l, col_m = st.columns(3)

with col_k:
    st.error(f"🔴 Risque ÉLEVÉ — {n_eleve} clients")
    st.markdown("""
    **Actions immédiates :**
    - 📞 Appel personnalisé sous 24h
    - 🎁 Offre contrat annuel (-20%)
    - 🔒 TechSupport + Security offerts 3 mois
    - 📊 Audit de satisfaction individuel
    """)

with col_l:
    st.warning(f"🟡 Risque MOYEN — {n_moyen} clients")
    st.markdown("""
    **Actions préventives :**
    - 📧 Email enquête de satisfaction
    - 🏆 Programme de fidélité
    - 💰 Révision tarif si charges > 70$/mois
    - 📱 Notification push app mobile
    """)

with col_m:
    st.success(f"🟢 Risque FAIBLE — {n_faible} clients")
    st.markdown("""
    **Maintien de la relation :**
    - 📰 Newsletter mensuelle personnalisée
    - ⭐ Valorisation de l'ancienneté client
    - 🔄 Upsell services complémentaires
    - 😊 Enquête NPS trimestrielle
    """)

st.divider()
st.caption("📊 Dashboard Analytique Churn | 4IASD 2025-2026 | XGBoost & CatBoost")