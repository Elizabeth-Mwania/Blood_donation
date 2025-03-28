from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# Define the input data model
class DonorInput(BaseModel):
    Age: int
    Genre: str
    Taille: float
    Poids: float
    Niveau_d_etude: str
    Situation_Matrimoniale_SM: str
    Profession: str
    Arrondissement_de_residence: str
    Quartier_de_Residence: str
    Nationalite: str
    Religion: str
    A_t_il_elle_deja_donne_le_sang: str
    Si_oui_preciser_la_date_du_dernier_don: str
    Taux_dhemoglobine: float
    Porteur_HIV_hbs_hcv: str
    Drepanocytaire: str
    Diabetique: str
    Hypertendus: str
    Asthmatiques: str
    Cardiaque: str
    Tatoue: str
    Scarifie: str
    La_DDR_est_mauvais_si_14_jour_avant_le_don: str
    Allaitement: str
    A_accoucher_ces_6_derniers_mois: str
    Interruption_de_grossesse_ces_06_derniers_mois: str

app = FastAPI()

# Load model artifacts
try:
    model = joblib.load("models/blood_donation_model.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    hemoglobin_bin_edges = joblib.load("models/hemoglobin_bin_edges.joblib")
except Exception as e:
    raise Exception(f"Error loading model or artifacts: {str(e)}")

def check_ineligibility_reasons(data: DonorInput):
    reasons = []
    input_dict = data.dict()
    
    # Permanent conditions (Severity 5)
    permanent_conditions = {
        "Porteur_HIV_hbs_hcv": "HIV/Hepatitis carrier",
        "Drepanocytaire": "Sickle cell disease",
        "Diabetique": "Diabetes",
        "Hypertendus": "Hypertension",
        "Asthmatiques": "Asthma",
        "Cardiaque": "Heart condition"
    }
    for field, reason in permanent_conditions.items():
        if input_dict[field] == "Oui":
            reasons.append({"reason": reason, "severity": 5, "type": "Permanent"})
    
    # Tattoo/Scarification (Severity 4)
    if input_dict["Tatoue"] == "Oui":
        reasons.append({"reason": "Recent tattoo", "severity": 4, "type": "Temporary"})
    if input_dict["Scarifie"] == "Oui":
        reasons.append({"reason": "Scarification", "severity": 4, "type": "Temporary"})
    
    # Hemoglobin levels (Severity 3)
    hb = input_dict["Taux_dhemoglobine"]
    if input_dict["Genre"] == "Femme" and hb < 12.5:
        reasons.append({"reason": "Hemoglobin below 12.5 g/dL (women)", "severity": 3, "type": "Temporary"})
    elif input_dict["Genre"] == "Homme" and hb < 13.0:
        reasons.append({"reason": "Hemoglobin below 13.0 g/dL (men)", "severity": 3, "type": "Temporary"})
    
    # Recent donation (Severity 3)
    if input_dict["A_t_il_elle_deja_donne_le_sang"] == "Oui":
        try:
            last_donation = datetime.strptime(input_dict["Si_oui_preciser_la_date_du_dernier_don"], "%Y-%m-%d")
            days_since = (datetime.now() - last_donation).days
            if days_since < 180:
                reasons.append({"reason": f"Donated {days_since} days ago (<180 days)", "severity": 3, "type": "Temporary"})
        except ValueError:
            pass
    
    # Female-specific conditions (Severity 3-4)
    if input_dict["Genre"] == "Femme":
        if input_dict["La_DDR_est_mauvais_si_14_jour_avant_le_don"] == "Oui":
            reasons.append({"reason": "Menstruation within 14 days", "severity": 3, "type": "Temporary"})
        if input_dict["Allaitement"] == "Oui":
            reasons.append({"reason": "Currently breastfeeding", "severity": 4, "type": "Temporary"})
        if input_dict["A_accoucher_ces_6_derniers_mois"] == "Oui":
            reasons.append({"reason": "Delivered in last 6 months", "severity": 4, "type": "Temporary"})
        if input_dict["Interruption_de_grossesse_ces_06_derniers_mois"] == "Oui":
            reasons.append({"reason": "Pregnancy termination in last 6 months", "severity": 4, "type": "Temporary"})
    
    # BMI (Severity 1-2)
    bmi = input_dict["Poids"] / (input_dict["Taille"] / 100) ** 2
    if bmi < 18.5:
        reasons.append({"reason": "BMI below 18.5 (Underweight)", "severity": 2, "type": "Temporary"})
    elif bmi >= 30:
        reasons.append({"reason": "BMI 30+ (Obese, may require checks)", "severity": 1, "type": "Warning"})
    
    # Age (Severity 1)
    if input_dict["Age"] < 19 or input_dict["Age"] > 65:
        reasons.append({"reason": "Age outside typical range (19-65)", "severity": 1, "type": "Warning"})
    
    reasons.sort(key=lambda x: x["severity"], reverse=True)
    return reasons

def preprocess_input(data: DonorInput):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    
    column_mapping = {
        "Niveau_d_etude": "Niveau d'etude",
        "Situation_Matrimoniale_SM": "Situation Matrimoniale (SM)",
        "Arrondissement_de_residence": "Arrondissement de residence",
        "Quartier_de_Residence": "Quartier de Residence",
        "A_t_il_elle_deja_donne_le_sang": "A-t-il (elle) deja donne le sang",
        "Si_oui_preciser_la_date_du_dernier_don": "Si oui preciser la date du dernier don.",
        "Taux_dhemoglobine": "Taux dhemoglobine",
        "Porteur_HIV_hbs_hcv": "Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]",
        "Drepanocytaire": "Raison de non-eligibilite totale  [Drepanocytaire]",
        "Diabetique": "Raison de non-eligibilite totale  [Diabetique]",
        "Hypertendus": "Raison de non-eligibilite totale  [Hypertendus]",
        "Asthmatiques": "Raison de non-eligibilite totale  [Asthmatiques]",
        "Cardiaque": "Raison de non-eligibilite totale  [Cardiaque]",
        "Tatoue": "Raison de non-eligibilite totale  [Tatoue]",
        "Scarifie": "Raison de non-eligibilite totale  [Scarifie]",
        "La_DDR_est_mauvais_si_14_jour_avant_le_don": "Raison de l'indisponibilite de la femme [La DDR est mauvais si <14 jour avant le don]",
        "Allaitement": "Raison de l'indisponibilite de la femme [Allaitement ]",
        "A_accoucher_ces_6_derniers_mois": "Raison de l'indisponibilite de la femme [A accoucher ces 6 derniers mois  ]",
        "Interruption_de_grossesse_ces_06_derniers_mois": "Raison de l'indisponibilite de la femme [Interruption de grossesse  ces 06 derniers mois]"
    }
    input_df = input_df.rename(columns=column_mapping)
    
    if input_df["Genre"].iloc[0] == "Homme":
        female_cols = [
            "Raison de l'indisponibilite de la femme [La DDR est mauvais si <14 jour avant le don]",
            "Raison de l'indisponibilite de la femme [Allaitement ]",
            "Raison de l'indisponibilite de la femme [A accoucher ces 6 derniers mois  ]",
            "Raison de l'indisponibilite de la femme [Interruption de grossesse  ces 06 derniers mois]"
        ]
        for col in female_cols:
            input_df[col] = "Non"
    
    binary_cols = [
        "A-t-il (elle) deja donne le sang", "Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]",
        "Raison de non-eligibilite totale  [Drepanocytaire]", "Raison de non-eligibilite totale  [Diabetique]",
        "Raison de non-eligibilite totale  [Hypertendus]", "Raison de non-eligibilite totale  [Asthmatiques]",
        "Raison de non-eligibilite totale  [Cardiaque]", "Raison de non-eligibilite totale  [Tatoue]",
        "Raison de non-eligibilite totale  [Scarifie]"
        # Note: Female-specific fields are not included here since the model doesn’t use them
    ]
    for col in binary_cols:
        input_df[col] = input_df[col].map({'Oui': 1, 'Non': 0})
    
    input_df['BMI'] = input_df['Poids'] / (input_df['Taille'] / 100) ** 2
    input_df['BMI_Category'] = pd.cut(input_df['BMI'], bins=[0, 18.5, 25, 30, float('inf')], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    input_df['Age_Group'] = pd.cut(input_df['Age'], bins=[0, 25, 35, 45, 55, float('inf')], labels=['Young', 'Young Adult', 'Middle Age', 'Senior Adult', 'Elderly'])
    input_df['Hemoglobin_Binned'] = pd.cut(input_df['Taux dhemoglobine'], bins=hemoglobin_bin_edges, labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
    
    def calculate_recent_donation(row):
        if row['A-t-il (elle) deja donne le sang'] == 1:
            try:
                last_donation_date = datetime.strptime(row['Si oui preciser la date du dernier don.'], '%Y-%m-%d')
                days_since = (datetime.now() - last_donation_date).days
                return 1 if days_since <= 180 else 0
            except ValueError:
                return 0
        return 0
    input_df['recent_donation'] = input_df.apply(calculate_recent_donation, axis=1)
    
    return input_df

@app.post("/predict")
async def predict(data: DonorInput):
    try:
        input_df = preprocess_input(data)
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)[0]
        probabilities = model.predict_proba(input_transformed)[0].tolist()
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        # Rule-based overrides for female-specific conditions
        ineligibility_reasons = check_ineligibility_reasons(data)
        if data.Genre == "Femme":
            if (data.Allaitement == "Oui" or 
                data.A_accoucher_ces_6_derniers_mois == "Oui" or 
                data.Interruption_de_grossesse_ces_06_derniers_mois == "Oui" or 
                data.La_DDR_est_mauvais_si_14_jour_avant_le_don == "Oui"):
                predicted_class = "Temporairement Non-eligible"
                # Adjust probabilities to emphasize temporary ineligibility
                probabilities = [0.1, 0.2, 0.7]  # [Definitive, Eligible, Temporary]
        
        # Generate ineligibility reasons if not eligible
        if predicted_class != "Eligible":
            ineligibility_reasons = check_ineligibility_reasons(data)
        else:
            ineligibility_reasons = []
        
        return {
            "prediction": predicted_class,
            "probability": probabilities,
            "ineligibility_reasons": ineligibility_reasons
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "API is running"}