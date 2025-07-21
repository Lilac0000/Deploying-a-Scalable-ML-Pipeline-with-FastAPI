import pytest
import pandas as pd
from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    performance_on_categorical_slice,
)

@pytest.fixture
def data_and_model():
    """
    Loads census data, preprocesses it, trains the model,
    and returns necessary components for testing.
    """
    data = pd.read_csv("data/census.csv")

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = "salary"  # Target column in the dataset

    X, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label=label, training=True
    )
    model = train_model(X, y)

    return data, categorical_features, label, encoder, lb, model, X, y

def test_model_training_and_inference(data_and_model):
    """
    Test that inference returns predictions of correct length.
    """
    _, _, _, _, _, model, X, y = data_and_model
    preds = inference(model, X)
    assert len(preds) == len(y), "Prediction length should match label length"

def test_model_metrics_computation(data_and_model):
    """
    Test that compute_model_metrics returns values between 0 and 1.
    """
    _, _, _, _, _, model, X, y = data_and_model
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F1 score should be between 0 and 1"

def test_performance_on_data_slice(data_and_model):
    """
    Test performance metrics computation on a slice of the data.
    """
    data, categorical_features, label, encoder, lb, model, _, _ = data_and_model
    slice_value = data["workclass"].iloc[0]
    precision, recall, fbeta = performance_on_categorical_slice(
        data=data,
        column_name="workclass",
        slice_value=slice_value,
        categorical_features=categorical_features,
        label=label,
        encoder=encoder,
        lb=lb,
        model=model,
    )
    assert 0 <= precision <= 1, "Slice precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Slice recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "Slice F1 score should be between 0 and 1"
