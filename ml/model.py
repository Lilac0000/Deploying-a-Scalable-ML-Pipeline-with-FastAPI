import inspect
from ml.data import process_data

print("process_data is from:", inspect.getfile(process_data))
print("process_data signature:", inspect.signature(process_data))



import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    """
    Run model inferences and return the predictions.
    """
    preds = model.predict(X)
    return preds

def save_model(model, path):
    """
    Serializes model to a file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    """
    Loads pickle file from `path` and returns it.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes model metrics on a slice of data where data[column_name] == slice_value.
    """
    # Filter the data for the slice
    data_slice = data[data[column_name] == slice_value]

    # Process the slice (training=False, reuse encoder and lb)
    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Get predictions
    preds = inference(model, X_slice)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
