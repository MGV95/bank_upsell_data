from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class ShapValueVisualizer:
    """Class to train Random Forest model to calculate SHAP values."""

    def __init__(self):
        self.rf_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "random_state": 42,
            "min_samples_leaf": 50,
            "max_features": "sqrt",
        }

        self.rf_classifier = RandomForestClassifier(**self.rf_params)

    def train_and_create_shap_plots(
        self, x_features: pd.DataFrame, y_outcome: pd.Series
    ):
        """Trains RandomForest, makes predictions, and produces a summary plot of SHAP values.

        The SHAP values indicate the contributions made by individual features to the score.

        Args:
            x_features: Predictive feature matrix
            y_outcome: Binary target variable to predict
        """
        # Split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            x_features, y_outcome, test_size=0.3, random_state=1
        )

        # Fit random forest classifier using training component.
        self.rf_classifier.fit(x_train, y_train)

        # Use Shap Tree explainer to compute SHAP values for test set.
        explainer = shap.TreeExplainer(self.rf_classifier)
        shap_values = explainer.shap_values(x_test)

        # Save shap plot
        plt.figure()
        shap.summary_plot(shap_values[:, :, 1], x_test, plot_type="violin", show=False)
        current_file = Path(__file__).resolve()
        data_path = current_file.parent.parent / 'outputs' / 'shap-plot.png'
        plt.savefig(data_path, bbox_inches="tight", dpi=300)
        plt.close()
