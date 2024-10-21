# This code implements an API using FastAPI to serve predictions and explanations for a credit model. 
# The model utilizes a pre-trained pipeline and SHAP (SHapley Additive exPlanations) values to interpret predictions.

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np

# Load the complete pipeline which includes both preprocessing steps and the trained XGBoost model.
pipeline_path = "xgb_pipeline_tuned.pkl"
loaded_pipeline = joblib.load(pipeline_path)

# Load the SHAP explainer for interpreting the model's predictions.
explainer_path = "shap_explainer.pkl"
loaded_shap_explainer = joblib.load(explainer_path)

# Create an instance of the FastAPI to build the web API.
app = FastAPI()

# Load client data from a CSV file. This data includes the feature set required for predictions.
data_clients = pd.read_csv("test_feature_engineering_sample.csv")

# Define an endpoint to get client data by client ID.
@app.get("/client_data/{client_id}")
# Function to fetch client data when provided with a specific client ID.
def get_client_data(client_id: int):
    # Select rows where the client ID matches the provided value.
    client_data = data_clients[data_clients["SK_ID_CURR"] == float(client_id)]
    # If no data is found, return a 404 error.
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    # Return the client's data as JSON.
    return {"data": client_data.to_dict(orient="records")}

# Define an endpoint to make predictions for a given client ID.
@app.get("/prediction/{client_id}")
# Function that returns a prediction score and threshold for a particular client.
def get_prediction(client_id: int):
    # Extract the data for the specified client and drop the TARGET column if it exists.
    client_data = data_clients[data_clients["SK_ID_CURR"] == float(client_id)].drop(columns=["TARGET"], errors='ignore')
    # If no data is found for the client, return a 404 error.
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    # Use the loaded pipeline to predict probabilities.
    prediction_proba = loaded_pipeline.predict_proba(client_data.to_numpy())[:, 1][0]  # Extract the prediction probability for the positive class.
    # Retrieve the decision threshold used by the loaded model.
    threshold = loaded_pipeline.named_steps['model'].get_threshold()
    # Return the prediction probability and threshold as a response.
    return {"score": float(prediction_proba), "threshold": float(threshold)}

# Define an endpoint to obtain SHAP values, which explain the prediction for a specific client ID.
@app.get("/shap_values/{client_id}")
# Function that returns SHAP values for the specified client.
def get_shap_values(client_id: int):
    # Retrieve data for the client and drop the TARGET column if present.
    client_data = data_clients[data_clients["SK_ID_CURR"] == float(client_id)].drop(columns=["TARGET"], errors='ignore')
    # If the client data set is empty, return a 404 error.
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    # Compute SHAP values for the retrieved client data.
    client_shap_values = loaded_shap_explainer.shap_values(client_data.to_numpy())
    # Return the computed SHAP values.
    return {"shap_values": client_shap_values.tolist()}
