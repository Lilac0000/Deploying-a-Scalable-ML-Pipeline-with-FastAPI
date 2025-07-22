import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# ✅ TODO: Provide paths to saved models
base_dir = os.path.dirname(__file__)
model = load_model(os.path.join(base_dir, "model", "model.pkl"))
encoder = load_model(os.path.join(base_dir, "model", "encoder.pkl"))
lb = load_model(os.path.join(base_dir, "model", "lb.pkl"))  # Add this if your apply_label needs it


# ✅ TODO: Create a FastAPI app
app = FastAPI()

# ✅ TODO: GET request for welcome message
@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Welcome to the income classifier API!"}

# ✅ TODO: POST request for inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # ✅ TODO: Process data for inference
    data_processed, _, _, _ = process_data(
        data, categorical_features=cat_features, training=False, encoder=encoder
    )

    # ✅ TODO: Make prediction
    _inference = inference(model, data_processed)

    return {"result": apply_label(_inference)}


