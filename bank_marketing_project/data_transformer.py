import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class DataTransformer:
    """DataTransformer handles transformation of raw data into features for ML Classifier."""

    def __init__(self, binary_cols, one_hot_encode_cols, drop_cols, education_col):
        """Initialises DataTransformer class by specifying how columns are transformed."""

        self.binary_cols = binary_cols
        self.one_hot_encode_cols = one_hot_encode_cols
        self.drop_cols = drop_cols

        self.education_col = education_col

    @classmethod
    def from_bank_data_col_names(cls):
        """Class method to create an instance of this class using the column names from the data source."""
        binary_cols = ["default", "housing", "loan", "y"]
        one_hot_encode_cols = ["job", "marital", "contact", "poutcome"]
        drop_cols = ["day", "month", "duration"]
        education_col = "education"

        return cls(
            binary_cols=binary_cols,
            one_hot_encode_cols=one_hot_encode_cols,
            drop_cols=drop_cols,
            education_col=education_col
        )

    def transform(self, raw_data: pd.DataFrame):
        """Performs operation that converts the raw data to a feature matrix.

        Low cardinality categorical variables are one-hot encoded, binary variables
        are mapped to a value in the set {0, 1}, some columns are excluded, and others
        undergo more complex transformations.

        Args:
            raw_data: DataFrame containing raw bank marketing campaign data.

        Returns:
            Feature matrix of transformed data.
        """
        for bin_col in self.binary_cols:
            raw_data[bin_col] = raw_data[bin_col].map({"yes": 1, "no": 0})

        raw_data[self.education_col] = self.education_to_ordinal(
            raw_data[self.education_col]
        )

        concat_df = self.one_hot_encode(raw_data[self.one_hot_encode_cols])
        raw_data = raw_data.drop(columns=self.one_hot_encode_cols)
        raw_data = pd.concat([raw_data, concat_df], axis=1)

        raw_data = raw_data.drop(columns=self.drop_cols)

        return raw_data

    def one_hot_encode(self, cat_df: pd.DataFrame) -> pd.DataFrame:
        """Function for one-hot encoding of low-cardinality categorical variables.

        The function returns a dataframe of one-hot encoded values. For each column in
        the provided cat_df, a separate column will be produced for each value of that column,
        which contains binary values indicating if the original column took the value listed
        in the name of the new column. Explicitly, column_1 containing values A, B and C will
        be mapped to columns column_1_A, column_1_B and column_1_C, which are binary columns
        that will take value 1 if the original column_1 took the corresponding value, and
        will be 0 otherwise.

        Args:
            cat_df: Dataframe containing only the categorical variables to be
                one-hot encoded.

        Returns:
            Dataframe of one-hot encoded categorical variables.
        """
        # Create an fit one-hot encoder.
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        one_hot_encoder.fit(cat_df)

        # Apply one-hot encoding transformation to categorical variables.
        one_hot_cols = one_hot_encoder.transform(cat_df)
        one_hot_col_names = one_hot_encoder.get_feature_names_out(self.one_hot_encode_cols)
        one_hot_df = pd.DataFrame(
            one_hot_cols, columns=one_hot_col_names, index=cat_df.index
        )
        return one_hot_df

    def education_to_ordinal(self, education_col: pd.Series) -> pd.Series:
        """Maps education category to a hierarchical level.

        Education level is mapped to ordinal integers, with higher integers corresponding
        to higher levels of education. The mappings are as follows:
         - Tertiary: 3
         - Secondary: 2
         - Primary: 1
         - Unknown: 0

        Args:
            education_col: Pandas series containing education level as a string value.

        Returns:
            Pandas series of education levels encoded as integers.
        """
        education_col = education_col.map(
            {"tertiary": 3, "secondary": 2, "primary": 1, "unknown": 0}
        )
        return education_col
