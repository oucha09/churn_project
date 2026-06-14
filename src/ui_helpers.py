def get_risk_style(probability):
    """Return the semantic label and colors associated with a churn score."""
    if probability >= 0.6:
        return {"label": "Risque eleve", "color": "#E24B4A", "bg": "#FCEBEB"}
    if probability >= 0.3:
        return {"label": "Risque moyen", "color": "#BA7517", "bg": "#FAEEDA"}
    return {"label": "Risque faible", "color": "#639922", "bg": "#EAF3DE"}


def risk_badge_html(probability):
    style = get_risk_style(probability)
    return (
        f'<span class="risk-badge" style="color:{style["color"]};'
        f'background:{style["bg"]};">{style["label"]}</span>'
    )


def result_card_html(prediction_label, probability):
    style = get_risk_style(probability)
    return f"""
    <div class="result-card">
        <div>
            <p class="eyebrow">Prediction</p>
            <p class="prediction-label" style="color:{style["color"]};">
                {prediction_label}
            </p>
        </div>
        <div class="probability">
            <p class="eyebrow">Probabilite de churn</p>
            <p class="probability-value">{probability:.0%}</p>
        </div>
    </div>
    """
