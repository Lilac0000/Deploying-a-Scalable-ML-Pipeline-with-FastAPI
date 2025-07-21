# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classification model trained to predict whether an individual's annual income exceeds $50,000 based on census demographic data. The model uses one-hot encoded categorical features and a supervised learning algorithm trained on processed census data.

## Intended Use

The model is intended to assist in socio-economic analyses by predicting income category (>50K or <=50K) from demographic features such as workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Training Data

The training data consists of census records containing demographic features and income labels. An 80/20 train/test split was applied. Categorical features were one-hot encoded, and labels binarized for training. The model was trained on this processed dataset.

## Evaluation Data

The evaluation was conducted on the held-out 20% test dataset from the original census data split. The test data was processed using the same encoder and label binarizer fitted on the training data to ensure consistent feature representation.

## Metrics
The model was evaluated using the following performance metrics on the test set:

Precision: 0.7419

Recall: 0.6384

F1 Score: 0.6863

These metrics reflect the model’s ability to correctly identify individuals earning over $50K while balancing false positives and false negatives.

## Ethical Considerations

The model relies on demographic features that may be correlated with sensitive attributes such as race and gender. Users must be cautious to avoid discriminatory practices or reinforcing existing social biases. The model should not be used for high-stakes decisions without additional fairness and bias mitigation analysis.

## Caveats and Recommendations
The model is trained on historical census data and may not generalize to other populations or time periods.

Performance may vary with data distribution shifts or missing feature values.

Users should validate the model on their specific data before deployment.

Additional fairness audits are recommended prior to use in sensitive contexts.
