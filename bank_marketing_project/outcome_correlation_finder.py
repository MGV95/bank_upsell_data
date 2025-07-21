from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


class OutcomeCorrelationFinder:
    """Class for training a decision tree with pre-specified parameters.

    The purpose of this tree is to be shallow enough to be readily inspected so
    that insights can be drawn into how different variables are contributing to
    likely outcomes.
    """

    def __init__(self):
        self.criterion = "entropy"
        self.max_depth = 4
        self.min_samples_leaf = 50
        self.dt_classifier = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

    def fit_decision_tree(self, X_features: pd.DataFrame, y_outcome: pd.Series):
        """Fits a shallow decision tree using provided data.

        Args:
            X_features: Design feature matrix containing variables used to predict outcome.
            y_outcome: Vector of binary target variables.
        """
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_outcome, test_size=0.3, random_state=1
        )

        # Train Decision Tree Classifier
        self.dt_classifier.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = self.dt_classifier.predict_proba(X_test)

    def save_decision_tree_representation(self, feature_names: List, plot_name: str):
        """Creates a visual representation of a decision tree and saves it in output location.

        Args:
            feature_names: List of feature names in feature matrix.
            plot_name: Name of saved plot image.
        """

        plt.figure(figsize=(12, 6))
        plot_tree(
            self.dt_classifier,
            feature_names=feature_names,
            class_names=['No', 'Yes'],
            filled=True,
            rounded=True,
            proportion=False,  # Shows sample counts instead of proportions
            impurity=True
        )
        plt.title(plot_name)
        current_file = Path(__file__).resolve()
        data_path = current_file.parent.parent / 'outputs' / 'Bank_Marketing_Decision_Tree.png'
        plt.savefig(data_path, bbox_inches="tight", dpi=300)
        plt.close()

