from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
import shap
import uvicorn

# Charger le modèle XGBoost et l'explainer SHAP à partir des fichiers enregistrés
with open('models/xgboost_classifier.pckl', 'rb') as f:
    model = pickle.load(f)

with open('models/xgboost_shap_explainer.pckl', 'rb') as f:
    explainer = pickle.load(f)

# Charge les données des clients à partir d'un fichier CSV
client_data = pd.read_csv('data/processed/test_feature_engineering.csv')

# Initialiser FastAPI pour créer l'API
app = FastAPI()

# 1. Route pour faire une prédiction à partir de SK_CURRENT_ID
"""Route /predict :

Input : SK_CURRENT_ID du client.
Output : La prédiction du modèle XGBoost (si le client est susceptible de rembourser ou non).
Fonctionnement : Le modèle prédit à partir des données du client en excluant SK_CURRENT_ID."""

@app.get("/predict")
def predict(SK_CURRENT_ID: int):
    # Recherche les données du client dans le DataFrame à partir de SK_CURRENT_ID
    client_row = client_data[client_data['SK_CURRENT_ID'] == SK_CURRENT_ID]
    
    # Si les données du client n'existent pas, retourne une erreur 404
    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client not found")

    #  prédiction en utilisant le modèle chargé (en excluant la colonne SK_CURRENT_ID)
    prediction = model.predict(client_row.drop(columns=['SK_CURRENT_ID']))[0]
    
    # Retourner l'ID du client et la prédiction sous forme de JSON
    return {"SK_CURRENT_ID": SK_CURRENT_ID, "prediction": int(prediction)}

# 2. Route pour retourner les données du client à partir de SK_CURRENT_ID

"""Route /client_data :

Input : SK_CURRENT_ID du client.
Output : Les données brutes du client sous forme de dictionnaire JSON.
Fonctionnement : Renvoie les données complètes pour un client particulier."""

@app.get("/client_data")
def get_client_data(SK_CURRENT_ID: int):
    # Recherche les données du client dans le DataFrame
    client_row = client_data[client_data['SK_CURRENT_ID'] == SK_CURRENT_ID]
    
    # Si le client n'est pas trouvé, retourner une erreur 404
    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client not found")

    # Retourne les données du client sous forme de dictionnaire JSON
    return client_row.to_dict(orient='records')

# 3. Route pour obtenir les valeurs SHAP d'un client à partir de SK_CURRENT_ID
"""Route /shap_values :

Input : SK_CURRENT_ID du client.
Output : Les valeurs SHAP du client pour expliquer la prédiction du modèle.
Fonctionnement : Utilise l'explainer SHAP pour donner les contributions des features individuelles à la prédiction."""
@app.get("/shap_values")

def get_shap_values(SK_CURRENT_ID: int):
    # Recherche les données du client, en excluant la colonne SK_CURRENT_ID
    client_row = client_data[client_data['SK_CURRENT_ID'] == SK_CURRENT_ID].drop(columns=['SK_CURRENT_ID'])
    
    # Si le client n'est pas trouvé, retourner une erreur 404
    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Calcule les valeurs SHAP pour le client
    shap_values = explainer.shap_values(client_row)
    
    # Retourne les valeurs SHAP sous forme de liste JSON
    return {"SK_CURRENT_ID": SK_CURRENT_ID, "shap_values": shap_values.tolist()}

# Démarre le serveur FastAPI en mode local sur le port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
