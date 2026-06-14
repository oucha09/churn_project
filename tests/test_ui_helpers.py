import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ui_helpers import get_risk_style


def test_risk_style_uses_requested_thresholds():
    assert get_risk_style(0.29)["label"] == "Risque faible"
    assert get_risk_style(0.30)["label"] == "Risque moyen"
    assert get_risk_style(0.60)["label"] == "Risque eleve"
