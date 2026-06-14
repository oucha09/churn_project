import sys
from pathlib import Path

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Bonus"))

from hyperparameter_tuning import build_searches


def test_tuning_uses_randomized_and_grid_search():
    searches = build_searches(quick=True)

    assert isinstance(searches["xgboost"], RandomizedSearchCV)
    assert isinstance(searches["catboost"], RandomizedSearchCV)
    assert isinstance(searches["logistic_regression"], GridSearchCV)
