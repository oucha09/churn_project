import pandas as pd

from ui_helpers import get_risk_style


def risk_level(probability):
    return get_risk_style(probability)["label"]


def customer_segment(client, probability):
    """Identifie un segment metier prioritaire a partir du profil et du risque."""
    if probability >= 0.60 and client.get("MonthlyCharges", 0) >= 80:
        return "Forte valeur a risque"
    if probability >= 0.60 and client.get("tenure", 0) <= 12:
        return "Nouveau client vulnerable"
    if client.get("Contract") == "Month-to-month" and probability >= 0.30:
        return "Sans engagement a convertir"
    if probability >= 0.30:
        return "Client a surveiller"
    if client.get("tenure", 0) >= 48:
        return "Client fidele a valoriser"
    return "Client stable"


def retention_recommendation(client, probability):
    """Produit une recommandation de fidelisation personnalisee et actionnable."""
    segment = customer_segment(client, probability)
    actions = {
        "Forte valeur a risque": (
            "Appel prioritaire sous 24h, remise personnalisee et revue de facture."
        ),
        "Nouveau client vulnerable": (
            "Parcours d'accueil renforce, appel de satisfaction et assistance offerte."
        ),
        "Sans engagement a convertir": (
            "Proposer un contrat annuel avec remise et avantage de fidelite."
        ),
        "Client a surveiller": (
            "Envoyer une enquete de satisfaction et une offre ciblee."
        ),
        "Client fidele a valoriser": (
            "Valoriser l'anciennete et proposer un avantage exclusif."
        ),
        "Client stable": "Maintenir la relation standard et suivre la satisfaction.",
    }
    return actions[segment]


def enrich_with_retention_segments(df, probability_column="churn_probability"):
    """Ajoute risque, segment et recommandation a un DataFrame clients."""
    enriched = df.copy()
    enriched["risk_level"] = enriched[probability_column].apply(risk_level)
    enriched["customer_segment"] = enriched.apply(
        lambda row: customer_segment(row, row[probability_column]), axis=1
    )
    enriched["recommendation"] = enriched.apply(
        lambda row: retention_recommendation(row, row[probability_column]), axis=1
    )
    return enriched


def segment_summary(df, probability_column="churn_probability"):
    """Resume taille, risque moyen et revenu mensuel par segment."""
    return (
        df.groupby("customer_segment", observed=True)
        .agg(
            clients=("customer_segment", "size"),
            probabilite_moyenne=(probability_column, "mean"),
            revenu_mensuel=("MonthlyCharges", "sum"),
        )
        .sort_values("probabilite_moyenne", ascending=False)
        .reset_index()
    )
