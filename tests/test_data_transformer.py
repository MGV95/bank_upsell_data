from unittest import TestCase
import pandas as pd

from bank_marketing_project.data_transformer import DataTransformer


class TestDataTransformer(TestCase):

    def setUp(self):

        self.data_transformer = DataTransformer(
            binary_cols=["col_0"],
            one_hot_encode_cols=["col_1", "col_2"],
            drop_cols=["col_3"],
            education_col="education"
        )

        test_data = {
            "col_0": ["yes", "no", "yes"],
            "col_1": ["A", "B", "A"],
            "col_2": ["A", "B", "C"],
            "col_3": ["D", "D", "D"],
            "education": ["tertiary", "primary", "secondary"],
        }

        self.test_dataframe = pd.DataFrame(test_data)

    def test_one_hot_encoder(self):
        """Test that one hot encoder method correctly encodes categorical variables."""
        categorical_dataframe = self.test_dataframe[["col_1", "col_2"]]
        encoded_data = self.data_transformer.one_hot_encode(categorical_dataframe)

        # Expected output columns as lists
        col_1_A = [1.0, 0.0, 1.0]
        col_1_B = [0.0, 1.0, 0.0]
        col_2_A = [1.0, 0.0, 0.0]
        col_2_B = [0.0, 1.0, 0.0]
        col_2_C = [0.0, 0.0, 1.0]

        column_value_mapping = {
            "col_1_A": col_1_A,
            "col_1_B": col_1_B,
            "col_2_A": col_2_A,
            "col_2_B": col_2_B,
            "col_2_C": col_2_C,
        }

        # Verify column names are as expected.
        self.assertSetEqual(
            set(encoded_data.columns.to_list()),
            set(column_value_mapping.keys())
        )

        # Verify the one-hot encoded values for each column are as expected.
        for col_name, col_values in column_value_mapping.items():
            self.assertListEqual(
                encoded_data[col_name].to_list(),
                col_values,
            )

    def test_education(self):
        education_series = self.test_dataframe["education"]
        transformed_data = self.data_transformer.education_to_ordinal(education_series)

        expected_transformed_output = [3.0, 1.0, 2.0]

        self.assertListEqual(transformed_data.to_list(), expected_transformed_output)

    def test_transform(self):

        transformed_data = self.data_transformer.transform(self.test_dataframe)

        # Verify dropped_cols are removed.
        self.assertTrue("col_3" not in transformed_data.columns)

        # Verify binary columns successfully transformed.
        expected_binary_values = [1.0, 0.0, 1.0]
        self.assertListEqual(transformed_data["col_0"].to_list(), expected_binary_values)
