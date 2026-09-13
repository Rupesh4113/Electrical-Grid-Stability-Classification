"""Model factory and hyperparameter space definitions."""

from typing import Dict, Tuple

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config.config import PARAM_GRIDS, RANDOM_STATE
from src.preprocessing import build_pipeline


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, Tuple[Pipeline, dict]]:
    """Build candidate pipelines and associated hyperparameter grids for tuning.

    Models:
    1. Linear SVM (scaled, tuned on C)
    2. Polynomial SVM (scaled, tuned on C, degree, gamma)
    3. RBF SVM (scaled, tuned on C, gamma)
    4. K-Nearest Neighbors (scaled, tuned on n_neighbors, weights, p)
    5. Decision Tree (unscaled, tuned on max_depth, min_samples_split, min_samples_leaf, criterion)

    Parameters
    ----------
    random_state : int, default RANDOM_STATE
        Random state for reproducibility.

    Returns
    -------
    Dict[str, Tuple[Pipeline, dict]]
        Dictionary mapping model names to (pipeline, param_grid).
    """
    models: Dict[str, Tuple[Pipeline, dict]] = {
        "Linear SVM": (
            build_pipeline(
                classifier=SVC(kernel="linear", probability=True, random_state=random_state),
                scale_features=True,
                include_feature_engineering=True,
            ),
            PARAM_GRIDS["Linear SVM"],
        ),
        "Polynomial SVM": (
            build_pipeline(
                classifier=SVC(kernel="poly", probability=True, random_state=random_state),
                scale_features=True,
                include_feature_engineering=True,
            ),
            PARAM_GRIDS["Polynomial SVM"],
        ),
        "RBF SVM": (
            build_pipeline(
                classifier=SVC(kernel="rbf", probability=True, random_state=random_state),
                scale_features=True,
                include_feature_engineering=True,
            ),
            PARAM_GRIDS["RBF SVM"],
        ),
        "KNN": (
            build_pipeline(
                classifier=KNeighborsClassifier(),
                scale_features=True,
                include_feature_engineering=True,
            ),
            PARAM_GRIDS["KNN"],
        ),
        "Decision Tree": (
            build_pipeline(
                classifier=DecisionTreeClassifier(random_state=random_state),
                scale_features=False,
                include_feature_engineering=True,
            ),
            PARAM_GRIDS["Decision Tree"],
        ),
    }

    return models
