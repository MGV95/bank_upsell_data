# Bank Upsell Marketing

This project is designed to be a brief end-to-end program for deriving insights into data from a bank upsell campaign.

The project achieves this in two ways:
- Firstly, a shallow decision tree is trained on the data, and its structure is saved, so we can assess what variables have the greatest impact on the outcome. 
- Secondly, a Random Forest is trained, and the feature shap values calculated, to assess the contribution to the score arising from individudal features.

These approaches were chosen because they are robust to more features being added and to the addition of more data. 
The decision tree also generates insights which involve more than one feature. 

## Running the code

In order to run the code, please navigate to the root directory of the project, and run `pip install .` to install dependencies specified in the toml. 

The script can then be run by running `python -m bank_marketing_project.main`

## Proposed Extensions

This was done quickly to comply with the suggested time constraints, and as a consequence, there are multiple additions and improvements that could be made to the repo.
These include:
- Manual analysis - the approaches above were chosen because they can readily be applied to multiple features without needing to manually interrogate individual variables. However, further investigations of important individual variables would likely be valuable.
- General purpose analyser - another approach would be to write a general purpose analyser which looks at the correlation between input features and the output variable. This could be made general, by treating numerical and categorical variables separately.
- Logging - the script is currently devoid of logging and this should be added if productionised.
- Data Ingestion and Validation - currently, the script is read directly to a pandas dataframe. If I had additional time, I'd add a data validator to check the data conforms to the expected values and format. 
- Extensibility - For convenience, some strings are hardcoded. If this were productionised, it would be valuable to think further about whether the codebase could be made more extensible. 
- Testing - Tests are added for fiddly transformations in the data_transformer module, but further tests could be added if more time were available. 

## Presentation

A powerpoint is attached together with this code base. This is intended as a presentation to the leader of a marketing department, rather than an explanation of the code and the rationale behind it, which is instead described here in this Readme.