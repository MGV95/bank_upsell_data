from pathlib import Path
from typing import Tuple

import pandas as pd

from bank_marketing_project.data_transformer import DataTransformer
from bank_marketing_project.outcome_correlation_finder import OutcomeCorrelationFinder
from bank_marketing_project.shap_visualizer import ShapValueVisualizer

def get_feature_matrix() -> Tuple:
    """Ingest raw data, transform to form feature matrix and target vector.

    Returns:
        Feature design matrix and target variables.
    """
    # Download Data
    current_file = Path(__file__).resolve()
    data_path = current_file.parent.parent / 'raw_data' / 'bank-full.csv'
    bank_full_df = pd.read_csv(data_path, sep=";")

    # Transform data from raw form into a feature matrix.
    data_transformer = DataTransformer.from_bank_data_col_names()
    feature_matrix = data_transformer.transform(bank_full_df)

    # Separate into features and target variable
    feature_cols = [col for col in feature_matrix.columns if col != "y"]
    x_feature_matrix = feature_matrix[feature_cols]
    y_target = feature_matrix["y"]
    return x_feature_matrix, y_target


def main():
    """Execute bank marketing interpretability project end-to-end.

    This code obtains a numerical feature matrix from the raw_data and subsequently
    trains a decision tree and a random forest on the matrix. The decision tree
    can be inspected directly to produce insights, whilst the feature contributions
    for the Random Forest can be computed to infer the contributions of each feature, and
    therefore understand how they contributed to the classification of the data point.
    """
    feature_matrix, targets = get_feature_matrix()

    # Fit Decision Tree and save visual representation in outputs folder.
    correlation_finder = OutcomeCorrelationFinder()
    correlation_finder.fit_decision_tree(feature_matrix, targets)
    correlation_finder.save_decision_tree_representation(
        feature_matrix.columns, "Decision_Tree_Bank_Upsell_Marketing_Outcomes.png"
    )

    # Train Random Forest and perform inference on a test set.
    # Save visual representation of SHAP values to understand feature impact.
    shap_visualizer = ShapValueVisualizer()
    shap_visualizer.train_and_create_shap_plots(feature_matrix, targets)

if __name__ == "__main__":
    main()
