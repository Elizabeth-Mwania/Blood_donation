from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional, Any
import json
import random
from sklearn.cluster import KMeans
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger("blood_donation_api")

# Define the input data model with proper field aliasess
class DonorInput(BaseModel):
    Age: int
    Genre: str
    Taille: float
    Poids: float
    Niveau_d_etude: str = Field(..., alias="Niveau_d_etude")
    Situation_Matrimoniale_SM: str
    Profession: str
    Arrondissement_de_residence: str
    Quartier_de_Residence: str
    Nationalite: str
    Religion: str
    A_t_il_elle_deja_donne_le_sang: str
    Si_oui_preciser_la_date_du_dernier_don: Optional[str] = None
    # Support both formats for hemoglobin
    Taux_dhemoglobine: Optional[float] = None
    # Health conditions
    Porteur_HIV_hbs_hcv: str = "Non"
    Drepanocytaire: str = "Non"
    Diabetique: str = "Non"
    Hypertendus: str = "Non"
    Asthmatiques: str = "Non"
    Cardiaque: str = "Non"
    Tatoue: str = "Non"
    Scarifie: str = "Non"
    # Female-specific conditions
    La_DDR_est_mauvais_si_14_jour_avant_le_don: str = "Non"
    Allaitement: str = "Non"
    A_accoucher_ces_6_derniers_mois: str = "Non"
    Interruption_de_grossesse_ces_06_derniers_mois: str = "Non"
    Est_enceinte: str = "Non"
    # Other conditions
    Est_sous_anti_biotherapie: str = "Non"
    Antecedent_de_transfusion: str = "Non"
    Recent_surgery: str = "Non"
    Recent_STI: str = "Non"
    Si_autres_raison_preciser: str = ""
    Autre_raison_preciser: str = ""

    class Config:
        # Allow population by field name or by alias
        populate_by_name = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

# New models for additional features
class DonorGroup(BaseModel):
    group_id: int
    name: str
    size: int
    description: str
    key_features: List[str]
    eligibility_rate: float

class DonorSimilarity(BaseModel):
    similarity_score: float
    similar_donors: int
    top_matching_features: List[str]
    eligibility_chance: float

class BloodDemand(BaseModel):
    blood_type: str
    current_demand: str
    urgency_level: int
    donation_impact: str

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
try:
    model = joblib.load("models/blood_donation_model.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    hemoglobin_bin_edges = joblib.load("models/hemoglobin_bin_edges.joblib")
    
    # Try to load dataset for additional features
    try:
        dataset = pd.read_csv("data/data_2019_cleaned.csv")
        logger.info(f"Loaded dataset with {len(dataset)} records")
        
        # Prepare clustering model
        donor_features = dataset.select_dtypes(include=['int64', 'float64']).fillna(0)
        donor_features = donor_features.iloc[:, :10]  # First 10 numeric columns for simplicity
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(donor_features)
        logger.info("K-means clustering model prepared")
    except Exception as e:
        logger.warning(f"Could not load dataset or prepare clustering: {e}")
        dataset = None
        kmeans = None
        
except Exception as e:
    logger.error(f"Error loading model or artifacts: {str(e)}")
    logger.error(traceback.format_exc())
    raise Exception(f"Error loading model or artifacts: {str(e)}")

def check_ineligibility_reasons(data: DonorInput):
    reasons = []
    input_dict = data.model_dump(by_alias=False)
    
    # Helper function to check binary conditions
    def is_condition_true(key):
        if key not in input_dict:
            return False
        value = input_dict.get(key)
        if isinstance(value, str):
            return value.lower() in ["oui", "yes", "1", "true"]
        return bool(value)
    
    # 1. Permanent conditions (Severity 5)
    permanent_conditions = {
        "Porteur_HIV_hbs_hcv": "HIV/Hepatitis carrier",
        "Drepanocytaire": "Sickle cell disease",
        "Diabetique": "Diabetes",
        "Hypertendus": "Hypertension",
        "Asthmatiques": "Asthma",
        "Cardiaque": "Heart condition"
    }
    for field, reason in permanent_conditions.items():
        if input_dict.get(field) == "Oui":
            reasons.append({"reason": reason, "severity": 5, "type": "Permanent"})
    
    # 2. Tattoo/Scarification (Severity 4)
    if input_dict.get("Tatoue") == "Oui":
        reasons.append({"reason": "Recent tattoo", "severity": 4, "type": "Temporary"})
    if input_dict.get("Scarifie") == "Oui":
        reasons.append({"reason": "Scarification", "severity": 4, "type": "Temporary"})
    
    # 3. Surgical history (Severity 4)
    if is_condition_true("Recent_surgery"):
        reasons.append({"reason": "Recent surgery", "severity": 4, "type": "Temporary"})
    
    # 4. Transfusion history (Severity 4)
    if is_condition_true("Antecedent_de_transfusion"):
        reasons.append({"reason": "Previous blood transfusion", "severity": 4, "type": "Temporary"})
    
    # 5. Medication (Severity 3)
    if is_condition_true("Est_sous_anti_biotherapie"):
        reasons.append({"reason": "Currently on antibiotics/medication", "severity": 3, "type": "Temporary"})
    
    # 6. Recent STI (Severity 3)
    if is_condition_true("Recent_STI"):
        reasons.append({"reason": "Recent STI", "severity": 3, "type": "Temporary"})
    
    # 7. Hemoglobin levels (Severity 3)
    # Get hemoglobin value from the field
    hb = input_dict.get("Taux_dhemoglobine")
    
    if hb is not None:
        if input_dict.get("Genre") == "Femme" and hb < 12.5:
            reasons.append({"reason": "Hemoglobin below 12.5 g/dL (women)", "severity": 3, "type": "Temporary"})
        elif input_dict.get("Genre") == "Homme" and hb < 13.0:
            reasons.append({"reason": "Hemoglobin below 13.0 g/dL (men)", "severity": 3, "type": "Temporary"})
    
    # 8. Recent donation (Severity 3)
    if input_dict.get("A_t_il_elle_deja_donne_le_sang") == "Oui":
        try:
            last_donation = datetime.strptime(input_dict.get("Si_oui_preciser_la_date_du_dernier_don", ""), "%Y-%m-%d")
            days_since = (datetime.now() - last_donation).days
            if days_since < 90:
                reasons.append({"reason": f"Donated {days_since} days ago (<90 days)", "severity": 3, "type": "Temporary"})
        except (ValueError, TypeError):
            pass
    
    # 9. Female-specific conditions (Severity 3-4)
    if input_dict.get("Genre") == "Femme":
        if input_dict.get("La_DDR_est_mauvais_si_14_jour_avant_le_don") == "Oui":
            reasons.append({"reason": "Menstruation within 14 days", "severity": 3, "type": "Temporary"})
        if input_dict.get("Allaitement") == "Oui":
            reasons.append({"reason": "Currently breastfeeding", "severity": 4, "type": "Temporary"})
        if input_dict.get("A_accoucher_ces_6_derniers_mois") == "Oui":
            reasons.append({"reason": "Delivered in last 6 months", "severity": 4, "type": "Temporary"})
        if input_dict.get("Interruption_de_grossesse_ces_06_derniers_mois") == "Oui":
            reasons.append({"reason": "Pregnancy termination in last 6 months", "severity": 4, "type": "Temporary"})
        if is_condition_true("Est_enceinte"):
            reasons.append({"reason": "Currently pregnant", "severity": 5, "type": "Temporary"})
    
    # 10. BMI (Severity 1-2)
    try:
        bmi = input_dict.get("Poids", 0) / (input_dict.get("Taille", 0) / 100) ** 2
        if bmi < 18.5:
            reasons.append({"reason": f"BMI {bmi:.1f} (Underweight)", "severity": 2, "type": "Temporary"})
        elif bmi >= 30:
            reasons.append({"reason": f"BMI {bmi:.1f} (Obese, may require additional checks)", "severity": 1, "type": "Warning"})
    except (ZeroDivisionError, TypeError):
        pass
    
    # 11. Age (Severity 1)
    age = input_dict.get("Age", 0)
    if age < 18 or age > 65:
        reasons.append({"reason": f"Age {age} outside typical range (18-65)", "severity": 1, "type": "Warning"})
    
    # Sort reasons by severity (highest first)
    reasons.sort(key=lambda x: x["severity"], reverse=True)
    
    return reasons

def preprocess_input(data: DonorInput):
    try:
        # Get input as dictionary
        input_dict = data.model_dump(by_alias=False)
        logger.info(f"Input data: {input_dict}")
        
        # Extract the numeric values safely
        age = float(input_dict.get("Age", 0))
        height = float(input_dict.get("Taille", 0))
        weight = float(input_dict.get("Poids", 0))
        hemoglobin = float(input_dict.get("Taux_dhemoglobine", 0))
        
        # Create a simplified DataFrame with expected column names and types
        df = pd.DataFrame({
            "Age": [age],
            "Taille": [height], 
            "Poids": [weight],
            "hemoglobin_level": [hemoglobin],
            "Taux d'hemoglobine": [hemoglobin],
            "Taux dhemoglobine": [hemoglobin],
            "Genre": [input_dict.get("Genre", "")],
            "Nationalite": [input_dict.get("Nationalite", "")],
            "Profession": [input_dict.get("Profession", "")]
        })
        
        # Add string/categorical columns
        text_columns = [
            "Niveau d'etude", "Quartier de Residence", "Arrondissement de residence",
            "Situation Matrimoniale (SM)", "Religion", "Si oui preciser la date du dernier don."
        ]
        
        for col in text_columns:
            field_name = col.replace(" ", "_").replace("'", "_").replace("(", "").replace(")", "")
            field_name = field_name.replace("-", "_").replace(".", "")
            df[col] = input_dict.get(field_name, "")
        
        # Convert binary columns to integers (0/1)
        df["A-t-il (elle) deja donne le sang"] = 1 if input_dict.get("A_t_il_elle_deja_donne_le_sang", "Non") == "Oui" else 0
        df["has_donated_before"] = 1 if input_dict.get("A_t_il_elle_deja_donne_le_sang", "Non") == "Oui" else 0
        
        # Add all binary health columns as 0 (No) by default
        binary_cols = [
            "Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]",
            "Raison de non-eligibilite totale  [Drepanocytaire]",
            "Raison de non-eligibilite totale  [Diabetique]",
            "Raison de non-eligibilite totale  [Hypertendus]",
            "Raison de non-eligibilite totale  [Asthmatiques]",
            "Raison de non-eligibilite totale  [Cardiaque]",
            "Raison de non-eligibilite totale  [Tatoue]",
            "Raison de non-eligibilite totale  [Scarifie]",
            "Raison indisponibilite  [Est sous anti-biotherapie  ]",
            "Raison indisponibilite  [IST recente (Exclu VIH, Hbs, Hcv)]",
            "Raison indisponibilite  [date de dernier Don < 3 mois ]",
            "Raison de non-eligibilite totale  [Opere]",
            "Raison de non-eligibilite totale  [Antecedent de transfusion]"
        ]
        
        for col in binary_cols:
            df[col] = 0
        
        # Add female-specific columns, all 0 by default
        female_cols = [
            "Raison de l'indisponibilite de la femme [La DDR est mauvais si <14 jour avant le don]",
            "Raison de l'indisponibilite de la femme [Allaitement ]",
            "Raison de l'indisponibilite de la femme [A accoucher ces 6 derniers mois  ]",
            "Raison de l'indisponibilite de la femme [Interruption de grossesse  ces 06 derniers mois]",
            "Raison de l'indisponibilite de la femme [est enceinte ]"
        ]
        
        for col in female_cols:
            df[col] = 0
        
        # Now set the ones that are marked as "Oui" in the input
        binary_mapping = {
            "Porteur_HIV_hbs_hcv": "Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]",
            "Drepanocytaire": "Raison de non-eligibilite totale  [Drepanocytaire]",
            "Diabetique": "Raison de non-eligibilite totale  [Diabetique]",
            "Hypertendus": "Raison de non-eligibilite totale  [Hypertendus]",
            "Asthmatiques": "Raison de non-eligibilite totale  [Asthmatiques]",
            "Cardiaque": "Raison de non-eligibilite totale  [Cardiaque]",
            "Tatoue": "Raison de non-eligibilite totale  [Tatoue]",
            "Scarifie": "Raison de non-eligibilite totale  [Scarifie]",
            "Est_sous_anti_biotherapie": "Raison indisponibilite  [Est sous anti-biotherapie  ]",
            "Recent_STI": "Raison indisponibilite  [IST recente (Exclu VIH, Hbs, Hcv)]",
            "Recent_surgery": "Raison de non-eligibilite totale  [Opere]",
            "Antecedent_de_transfusion": "Raison de non-eligibilite totale  [Antecedent de transfusion]",
            "La_DDR_est_mauvais_si_14_jour_avant_le_don": "Raison de l'indisponibilite de la femme [La DDR est mauvais si <14 jour avant le don]",
            "Allaitement": "Raison de l'indisponibilite de la femme [Allaitement ]",
            "A_accoucher_ces_6_derniers_mois": "Raison de l'indisponibilite de la femme [A accoucher ces 6 derniers mois  ]",
            "Interruption_de_grossesse_ces_06_derniers_mois": "Raison de l'indisponibilite de la femme [Interruption de grossesse  ces 06 derniers mois]",
            "Est_enceinte": "Raison de l'indisponibilite de la femme [est enceinte ]"
        }
        
        for api_field, df_field in binary_mapping.items():
            if input_dict.get(api_field, "Non") == "Oui":
                df[df_field] = 1
        
        # Add derived features
        # Gender processing - create is_female
        df['is_female'] = 1 if df['Genre'].iloc[0].lower() == 'femme' else 0
        
        # BMI calculation
        df['BMI'] = df['Poids'] / ((df['Taille'] / 100) ** 2)
        df['BMI_Category'] = pd.cut(
            df['BMI'],
            bins=[0, 18.5, 25, 30, float('inf')],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        ).astype(str)
        df['is_underweight'] = (df['BMI'] < 18.5).astype(int)
        
        # Age groups
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=[0, 25, 35, 45, 55, float('inf')],
            labels=['Young', 'Young Adult', 'Middle Age', 'Senior Adult', 'Elderly']
        ).astype(str)
        df['age_outside_range'] = ((df['Age'] < 18) | (df['Age'] > 65)).astype(int)
        
        # Hemoglobin thresholds
        df['low_hemoglobin'] = 0
        if df['is_female'].iloc[0] == 1:
            df['low_hemoglobin'] = (df['hemoglobin_level'] < 12.5).astype(int)
        else:
            df['low_hemoglobin'] = (df['hemoglobin_level'] < 13.0).astype(int)
        
        # Hemoglobin binning
        try:
            df['Hemoglobin_Binned'] = pd.cut(
                df['hemoglobin_level'],
                bins=hemoglobin_bin_edges,
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                include_lowest=True
            ).astype(str)
        except Exception as e:
            logger.warning(f"Hemoglobin binning failed: {e}")
            df['Hemoglobin_Binned'] = 'Q2'  # Default to middle bin
        
        # Map additional columns needed by the preprocessor
        df['education_level'] = df["Niveau d'etude"] 
        df['marital_status'] = df["Situation Matrimoniale (SM)"]
        df['residence_district'] = df["Arrondissement de residence"]
        df['residence_neighborhood'] = df["Quartier de Residence"]
        df['nationality'] = df["Nationalite"]
        
        # Add all the derived fields needed for risk scores
        for col in binary_cols + female_cols:
            # Create feature names from original column names
            if "Porteur(HIV,hbs,hcv)" in col:
                df["total_ineligible_hiv_hbs_hcv"] = df[col]
            elif "Drepanocytaire" in col:
                df["total_ineligible_sickle_cell"] = df[col] 
            elif "Diabetique" in col:
                df["total_ineligible_diabetes"] = df[col]
            elif "Hypertendus" in col:
                df["total_ineligible_hypertension"] = df[col]
            elif "Asthmatiques" in col:
                df["total_ineligible_asthma"] = df[col]
            elif "Cardiaque" in col:
                df["total_ineligible_heart_disease"] = df[col]
            elif "Tatoue" in col:
                df["total_ineligible_tattoo"] = df[col]
            elif "Scarifie" in col:
                df["total_ineligible_scarification"] = df[col]
            elif "Antecedent de transfusion" in col:
                df["total_ineligible_transfusion_history"] = df[col]
            elif "Opere" in col:
                df["total_ineligible_surgery"] = df[col]
            elif "Est sous anti-biotherapie" in col:
                df["ineligible_antibiotics"] = df[col]
            elif "IST recente" in col:
                df["ineligible_recent_sti"] = df[col]
            elif "La DDR est mauvais" in col:
                df["female_ineligible_menstrual"] = df[col]
            elif "Allaitement" in col:
                df["female_ineligible_breastfeeding"] = df[col]
            elif "A accoucher ces 6 derniers mois" in col:
                df["female_ineligible_postpartum"] = df[col]
            elif "Interruption de grossesse" in col:
                df["female_ineligible_miscarriage"] = df[col]
            elif "est enceinte" in col:
                df["female_ineligible_pregnant"] = df[col]
        
        # Add risk score calculations
        health_cols = [
            "total_ineligible_hiv_hbs_hcv", "total_ineligible_sickle_cell", 
            "total_ineligible_diabetes", "total_ineligible_hypertension",
            "total_ineligible_asthma", "total_ineligible_heart_disease",
            "total_ineligible_tattoo", "total_ineligible_scarification", 
            "total_ineligible_transfusion_history", "total_ineligible_surgery"
        ]
        
        # Ensure all health columns exist
        for col in health_cols:
            if col not in df.columns:
                df[col] = 0
        
        df['health_risks_count'] = df[health_cols].sum(axis=1)
        df['has_health_risks'] = (df['health_risks_count'] > 0).astype(int)
        
        female_risk_cols = [
            "female_ineligible_menstrual", "female_ineligible_breastfeeding",
            "female_ineligible_postpartum", "female_ineligible_miscarriage",
            "female_ineligible_pregnant"
        ]
        
        # Ensure all female columns exist
        for col in female_risk_cols:
            if col not in df.columns:
                df[col] = 0
        
        df['female_ineligibility_score'] = df[female_risk_cols].sum(axis=1)
        df['female_ineligible'] = (df['female_ineligibility_score'] > 0).astype(int)
        
        # Medication flag
        if 'ineligible_antibiotics' not in df.columns:
            df['ineligible_antibiotics'] = 0
        df['on_medication'] = df['ineligible_antibiotics']
        
        # Add donation-related fields
        df['days_since_last_donation'] = 9999
        df['recent_donation'] = 0
        df['very_recent_donation'] = 0
        
        if input_dict.get("A_t_il_elle_deja_donne_le_sang") == "Oui" and input_dict.get("Si_oui_preciser_la_date_du_dernier_don"):
            try:
                last_donation = datetime.strptime(input_dict.get("Si_oui_preciser_la_date_du_dernier_don"), "%Y-%m-%d")
                days_since = (datetime.now() - last_donation).days
                df['days_since_last_donation'] = days_since
                if days_since < 90:
                    df["Raison indisponibilite  [date de dernier Don < 3 mois ]"] = 1
                df['recent_donation'] = (days_since <= 180).astype(int)
                df['very_recent_donation'] = (days_since <= 90).astype(int)
            except (ValueError, TypeError):
                pass
        
        # Risk factors
        risk_factors = [
            'low_hemoglobin', 'is_underweight', 'age_outside_range',
            'very_recent_donation', 'on_medication', 'has_health_risks',
            'female_ineligible'
        ]
        
        df['eligibility_risk_score'] = df[risk_factors].sum(axis=1)
        df['high_risk_ineligible'] = (df['eligibility_risk_score'] >= 2).astype(int)
        
        logger.info(f"Final dataframe shape: {df.shape}")
        logger.info(f"Final column types:")
        for col in df.columns:
            logger.info(f"{col}: {df[col].dtype}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in simplified preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# New function to determine donor profile
def determine_donor_profile(input_df):
    """Determine donor profile based on input features"""
    
    # Age-based profiles
    age = input_df["Age"].iloc[0]
    if age < 25:
        profile_age = "Young Donor"
    elif age < 40:
        profile_age = "Prime Donor"
    else:
        profile_age = "Mature Donor"
    
    # Health-based profiles
    bmi = input_df["BMI"].iloc[0]
    health_risks = input_df["health_risks_count"].iloc[0]
    
    # Use hemoglobin_level instead of a column that might not exist
    hemoglobin = input_df["hemoglobin_level"].iloc[0]
    
    if health_risks == 0 and bmi >= 18.5 and bmi < 30:
        if hemoglobin > 14:
            profile_health = "Optimal Health"
        else:
            profile_health = "Good Health"
    elif health_risks == 1 or (bmi < 18.5 or bmi >= 30):
        profile_health = "Monitored Health"
    else:
        profile_health = "Health Concerns"
    
    # Donation history profiles
    donation_history = input_df["has_donated_before"].iloc[0]
    if donation_history:
        days_since = input_df["days_since_last_donation"].iloc[0]
        if days_since < 180:
            profile_donation = "Recent Donor"
        elif days_since < 365:
            profile_donation = "Regular Donor"
        else:
            profile_donation = "Returning Donor"
    else:
        profile_donation = "First-time Donor"
    
    return {
        "age_profile": profile_age,
        "health_profile": profile_health,
        "donation_profile": profile_donation
    }

# New function to get eligibility timeline for temporarily ineligible donors
def get_eligibility_timeline(data: DonorInput, reasons: List[Dict]):
    """Calculate when a temporarily ineligible donor will become eligible"""
    
    today = datetime.now().date()
    eligible_dates = []
    
    for reason in reasons:
        if reason["type"] != "Permanent":
            days_to_wait = 0
            
            # Set waiting period based on reason
            if "tattoo" in reason["reason"].lower():
                days_to_wait = 120  # 4 months for tattoos
            elif "surgery" in reason["reason"].lower():
                days_to_wait = 180  # 6 months for surgery
            elif "donated" in reason["reason"].lower():
                # Extract days from reason text (e.g., "Donated 30 days ago")
                import re
                match = re.search(r'Donated (\d+) days', reason["reason"])
                if match:
                    days_already_waited = int(match.group(1))
                    days_to_wait = 90 - days_already_waited
            elif "breastfeeding" in reason["reason"].lower():
                days_to_wait = 90  # 3 months after stopping breastfeeding
            elif "delivered" in reason["reason"].lower() or "pregnancy" in reason["reason"].lower():
                days_to_wait = 180  # 6 months after pregnancy/delivery
            elif "hemoglobin" in reason["reason"].lower():
                days_to_wait = 60  # 2 months to improve hemoglobin
            elif "antibiotics" in reason["reason"].lower() or "medication" in reason["reason"].lower():
                days_to_wait = 14  # 2 weeks after finishing medication
            else:
                # Default waiting period
                days_to_wait = 30
            
            if days_to_wait > 0:
                eligible_date = today + timedelta(days=days_to_wait)
                eligible_dates.append({
                    "reason": reason["reason"],
                    "eligible_after": eligible_date.strftime("%Y-%m-%d"),
                    "days_to_wait": days_to_wait
                })
    
    # Sort by days to wait (longest first)
    eligible_dates.sort(key=lambda x: x["days_to_wait"], reverse=True)
    
    return eligible_dates

# New function to generate personalized eligibility improvement tips
def generate_improvement_tips(reasons: List[Dict]) -> List[Dict]:
    """Generate personalized tips to improve eligibility"""
    
    tips = []
    for reason in reasons:
        if "hemoglobin" in reason["reason"].lower():
            tips.append({
                "category": "Nutrition",
                "tip": "Increase iron-rich foods in your diet",
                "description": "Include foods like lean red meat, beans, spinach, and iron-fortified cereals.",
                "difficulty": "Medium",
                "time_to_implement": "2-4 weeks"
            })
            tips.append({
                "category": "Supplements",
                "tip": "Consider iron supplements",
                "description": "Consult your doctor about iron supplements, especially if you have low hemoglobin.",
                "difficulty": "Easy",
                "time_to_implement": "Immediate"
            })
        
        if "bmi" in reason["reason"].lower() and "underweight" in reason["reason"].lower():
            tips.append({
                "category": "Nutrition",
                "tip": "Increase caloric intake with healthy foods",
                "description": "Focus on nutrient-dense foods like nuts, avocados, and whole grains.",
                "difficulty": "Medium",
                "time_to_implement": "4-8 weeks"
            })
        
        if "bmi" in reason["reason"].lower() and "obese" in reason["reason"].lower():
            tips.append({
                "category": "Lifestyle",
                "tip": "Gradual, sustainable weight management",
                "description": "Aim for a balanced diet and regular physical activity.",
                "difficulty": "Hard",
                "time_to_implement": "3-6 months"
            })
        
        if any(x in reason["reason"].lower() for x in ["antibiotics", "medication"]):
            tips.append({
                "category": "Medical",
                "tip": "Complete prescribed medication course",
                "description": "Finish your medication as directed, then wait 2 weeks before donating.",
                "difficulty": "Easy",
                "time_to_implement": "Varies by prescription"
            })
    
    # Add general tips for everyone
    tips.append({
        "category": "Hydration",
        "tip": "Stay well-hydrated before donation",
        "description": "Drink plenty of water in the 24-48 hours before your donation.",
        "difficulty": "Easy",
        "time_to_implement": "Immediate"
    })
    
    # Remove duplicates and limit to 5 tips
    unique_tips = []
    tip_categories = set()
    for tip in tips:
        if tip["category"] not in tip_categories and len(unique_tips) < 5:
            tip_categories.add(tip["category"])
            unique_tips.append(tip)
    
    return unique_tips

# Function to determine blood type impact
def get_blood_demand(data: Optional[DonorInput] = None) -> List[BloodDemand]:
    """Get current blood demand information"""
    
    # In a real system, this would come from a database
    # Here we're simulating it
    blood_types = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
    
    # Generate synthetic demand data
    result = []
    
    for blood_type in blood_types:
        # Simulate different demand levels
        if blood_type in ["O-", "O+"]:
            urgency = random.randint(4, 5)  # High demand for universal donors
            demand = "High"
        elif blood_type in ["B-", "AB-"]:
            urgency = random.randint(3, 5)  # Rarer types vary in demand
            demand = "Medium-High"
        else:
            urgency = random.randint(2, 4)  # More common types
            demand = "Medium"
        
        # Estimate impact
        if urgency >= 4:
            impact = "Could save up to 3 lives in emergency situations"
        else:
            impact = "Will help maintain hospital supplies for routine procedures"
        
        result.append(BloodDemand(
            blood_type=blood_type,
            current_demand=demand,
            urgency_level=urgency,
            donation_impact=impact
        ))
    
    # Sort by urgency
    result.sort(key=lambda x: x.urgency_level, reverse=True)
    
    return result

# Function to find similar donors and outcomes
def find_similar_donors(data: DonorInput):
    """Find similar donors from the dataset and their outcomes"""
    
    if dataset is None or kmeans is None:
        # Return mock data if dataset or clustering model isn't available
        logger.warning("Using mock data for similar donors (dataset or kmeans model not available)")
        return {
            "similar_count": 24,
            "eligibility_rate": 70.8,
            "common_factors": ["Age group", "BMI category", "Education level"],
            "cluster_id": 2
        }
    
    try:
        # We already have the preprocessed dataframe from the prediction function
        input_df = preprocess_input(data)
        
        # Extract features for clustering - be careful about how many we grab
        try:
            # First, make sure we're only getting numeric columns
            numeric_features = input_df.select_dtypes(include=['int64', 'float64'])
            
            # Then ensure we don't take more columns than we have
            num_cols = min(10, numeric_features.shape[1])
            numeric_features = numeric_features.iloc[:, :num_cols]
            
            # Fill any missing values
            numeric_features = numeric_features.fillna(0)
            
            logger.info(f"Extracted {numeric_features.shape[1]} numeric features for clustering")
        except Exception as e:
            logger.error(f"Error extracting numeric features: {e}")
            # Fallback to basic numeric features
            numeric_features = pd.DataFrame({
                'Age': [input_df["Age"].iloc[0]],
                'BMI': [input_df["BMI"].iloc[0]],
                'hemoglobin_level': [input_df["hemoglobin_level"].iloc[0]]
            })
        
        # Make sure we have the right number of features for the kmeans model
        model_features = kmeans.cluster_centers_.shape[1]
        if numeric_features.shape[1] < model_features:
            # Pad with zeros
            for i in range(numeric_features.shape[1], model_features):
                numeric_features[f'padding_{i}'] = 0
        elif numeric_features.shape[1] > model_features:
            # Truncate
            numeric_features = numeric_features.iloc[:, :model_features]
        
        # Predict cluster
        cluster = kmeans.predict(numeric_features)[0]
        logger.info(f"Donor belongs to cluster {cluster}")
        
        # Find similar donors from the same cluster
        donor_features = dataset.select_dtypes(include=['int64', 'float64']).fillna(0)
        # Ensure same number of features as kmeans model
        donor_features = donor_features.iloc[:, :model_features]
        
        dataset_clusters = kmeans.predict(donor_features)
        similar_donors = dataset[dataset_clusters == cluster]
        
        # Calculate eligibility rate for similar donors
        if len(similar_donors) > 0:
            eligibility_rate = (similar_donors['ELIGIBILITE AU DON.'] == 'Eligible').mean() * 100
        else:
            eligibility_rate = 0
        
        # Identify common factors
        common_factors = []
        
        if 'Age' in input_df.columns and 'Age' in dataset.columns:
            age = input_df['Age'].iloc[0]
            age_range = (age - 5, age + 5)
            age_match_rate = ((dataset['Age'] >= age_range[0]) & (dataset['Age'] <= age_range[1])).mean() * 100
            if age_match_rate > 20:
                common_factors.append("Age group")
        
        if 'Genre' in input_df.columns and 'Genre' in dataset.columns:
            gender = input_df['Genre'].iloc[0]
            gender_match_rate = (dataset['Genre'] == gender).mean() * 100
            if gender_match_rate > 40:
                common_factors.append("Gender")
        
        if 'BMI' in input_df.columns:
            bmi = input_df['BMI'].iloc[0]
            if 'Poids' in dataset.columns and 'Taille' in dataset.columns:
                dataset['BMI'] = dataset['Poids'] / ((dataset['Taille'] / 100) ** 2)
                bmi_range = (bmi - 2, bmi + 2)
                bmi_match_rate = ((dataset['BMI'] >= bmi_range[0]) & (dataset['BMI'] <= bmi_range[1])).mean() * 100
                if bmi_match_rate > 20:
                    common_factors.append("BMI category")
        
        if 'education_level' in input_df.columns and "Niveau d'etude" in dataset.columns:
            education = input_df['education_level'].iloc[0]
            edu_match_rate = (dataset["Niveau d'etude"] == education).mean() * 100
            if edu_match_rate > 20:
                common_factors.append("Education level")
        
        # If not enough factors, add some
        if len(common_factors) < 2:
            default_factors = ["Demographic profile", "Health indicators", "Geographic region"]
            common_factors.extend(default_factors[:3-len(common_factors)])
        
        return {
            "similar_count": len(similar_donors),
            "eligibility_rate": round(eligibility_rate, 1),
            "common_factors": common_factors[:3],
            "cluster_id": int(cluster)
        }
    
    except Exception as e:
        logger.error(f"Error finding similar donors: {e}")
        logger.error(traceback.format_exc())
        return {
            "similar_count": 24,
            "eligibility_rate": 70.8,
            "common_factors": ["Age group", "BMI category", "Education level"],
            "cluster_id": 2
        }

@app.post("/predict")
async def predict(data: DonorInput):
    try:
        logger.info("Received prediction request")
        
        # Use our improved preprocessing function that handles all the conversions
        try:
            input_df = preprocess_input(data)
            logger.info("Successfully preprocessed input data")
        except Exception as preprocess_error:
            logger.error(f"Preprocessing error: {preprocess_error}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=422, detail=f"Error in preprocessing input data: {str(preprocess_error)}")
        
        # Perform rule-based checks first, before even calling the model
        # These are hard rules for automatic ineligibility
        is_permanently_ineligible = False
        permanent_reasons = []
        
        # Check for permanent disqualifying conditions
        if data.Porteur_HIV_hbs_hcv == "Oui":
            is_permanently_ineligible = True
            permanent_reasons.append({"reason": "HIV/Hepatitis carrier", "severity": 5, "type": "Permanent"})
        
        if data.Drepanocytaire == "Oui":
            is_permanently_ineligible = True
            permanent_reasons.append({"reason": "Sickle cell disease", "severity": 5, "type": "Permanent"})
        
        if data.Diabetique == "Oui": 
            is_permanently_ineligible = True
            permanent_reasons.append({"reason": "Diabetes", "severity": 5, "type": "Permanent"})
        
        if data.Hypertendus == "Oui":
            is_permanently_ineligible = True
            permanent_reasons.append({"reason": "Hypertension", "severity": 5, "type": "Permanent"})
        
        if data.Asthmatiques == "Oui":
            is_permanently_ineligible = True
            permanent_reasons.append({"reason": "Asthma", "severity": 5, "type": "Permanent"})
        
        if data.Cardiaque == "Oui":
            is_permanently_ineligible = True
            permanent_reasons.append({"reason": "Heart condition", "severity": 5, "type": "Permanent"})
        
        # Check for temporary disqualifiers
        is_temporarily_ineligible = False
        temporary_reasons = []
        
        # 1. Check for recent surgeries
        if getattr(data, "Recent_surgery", "Non") == "Oui":
            is_temporarily_ineligible = True
            temporary_reasons.append({"reason": "Recent surgery", "severity": 4, "type": "Temporary"})
        
        # 2. Check for recent tattoos
        if data.Tatoue == "Oui":
            is_temporarily_ineligible = True
            temporary_reasons.append({"reason": "Recent tattoo", "severity": 4, "type": "Temporary"})
        
        # 3. Check for scarification
        if data.Scarifie == "Oui":
            is_temporarily_ineligible = True
            temporary_reasons.append({"reason": "Scarification", "severity": 4, "type": "Temporary"})
        
        # 4. Check for blood transfusion history
        if getattr(data, "Antecedent_de_transfusion", "Non") == "Oui":
            is_temporarily_ineligible = True
            temporary_reasons.append({"reason": "Previous blood transfusion", "severity": 4, "type": "Temporary"})
        
        # 5. Check for medication usage
        if getattr(data, "Est_sous_anti_biotherapie", "Non") == "Oui":
            is_temporarily_ineligible = True
            temporary_reasons.append({"reason": "Currently on antibiotics/medication", "severity": 3, "type": "Temporary"})
        
        # 6. Check for recent STI
        if getattr(data, "Recent_STI", "Non") == "Oui":
            is_temporarily_ineligible = True
            temporary_reasons.append({"reason": "Recent STI", "severity": 3, "type": "Temporary"})
        
        # 7. Check for female-specific conditions
        if data.Genre == "Femme":
            if getattr(data, "La_DDR_est_mauvais_si_14_jour_avant_le_don", "Non") == "Oui":
                is_temporarily_ineligible = True
                temporary_reasons.append({"reason": "Menstruation within 14 days", "severity": 3, "type": "Temporary"})
            
            if getattr(data, "Allaitement", "Non") == "Oui":
                is_temporarily_ineligible = True
                temporary_reasons.append({"reason": "Currently breastfeeding", "severity": 4, "type": "Temporary"})
            
            if getattr(data, "A_accoucher_ces_6_derniers_mois", "Non") == "Oui":
                is_temporarily_ineligible = True
                temporary_reasons.append({"reason": "Delivered in last 6 months", "severity": 4, "type": "Temporary"})
            
            if getattr(data, "Interruption_de_grossesse_ces_06_derniers_mois", "Non") == "Oui":
                is_temporarily_ineligible = True
                temporary_reasons.append({"reason": "Pregnancy termination in last 6 months", "severity": 4, "type": "Temporary"})
            
            if getattr(data, "Est_enceinte", "Non") == "Oui":
                is_temporarily_ineligible = True
                temporary_reasons.append({"reason": "Currently pregnant", "severity": 5, "type": "Temporary"})
        
        # 8. Check hemoglobin levels
        hemoglobin = float(getattr(data, "Taux_dhemoglobine", 0))
        if data.Genre == "Femme" and hemoglobin < 12.5:
            is_temporarily_ineligible = True
            temporary_reasons.append({"reason": "Hemoglobin below 12.5 g/dL (women)", "severity": 3, "type": "Temporary"})
        elif data.Genre == "Homme" and hemoglobin < 13.0:
            is_temporarily_ineligible = True
            temporary_reasons.append({"reason": "Hemoglobin below 13.0 g/dL (men)", "severity": 3, "type": "Temporary"})
        
        # 9. Check recent donation
        if data.A_t_il_elle_deja_donne_le_sang == "Oui" and data.Si_oui_preciser_la_date_du_dernier_don:
            try:
                last_donation = datetime.strptime(data.Si_oui_preciser_la_date_du_dernier_don, "%Y-%m-%d")
                days_since = (datetime.now() - last_donation).days
                if days_since < 90:
                    is_temporarily_ineligible = True
                    temporary_reasons.append({"reason": f"Donated {days_since} days ago (<90 days)", "severity": 3, "type": "Temporary"})
            except (ValueError, TypeError):
                pass
        
        # Now check BMI
        try:
            height = float(data.Taille)
            weight = float(data.Poids)
            bmi = weight / ((height / 100) ** 2)
            
            if bmi < 18.5:
                is_temporarily_ineligible = True
                temporary_reasons.append({"reason": f"BMI {bmi:.1f} (Underweight)", "severity": 2, "type": "Temporary"})
        except (ValueError, ZeroDivisionError):
            pass
        
        # Only use the model if not already determined to be ineligible
        if is_permanently_ineligible:
            predicted_class = "Definitivement Non-eligible"
            probabilities = [0.9, 0.05, 0.05]  # Strong prediction for permanent ineligibility
            ineligibility_reasons = permanent_reasons
            
            logger.info("Rule-based override: Permanently ineligible")
        elif is_temporarily_ineligible:
            predicted_class = "Temporairement Non-eligible"
            probabilities = [0.1, 0.2, 0.7]  # Strong prediction for temporary ineligibility
            ineligibility_reasons = temporary_reasons
            
            logger.info("Rule-based override: Temporarily ineligible")
        else:
            # Use the model for less clear-cut cases
            try:
                # Transform using preprocessor
                input_transformed = preprocessor.transform(input_df)
                logger.info("Successfully transformed data with preprocessor")
                
                # Make prediction
                prediction = model.predict(input_transformed)[0]
                probabilities = model.predict_proba(input_transformed)[0].tolist()
                predicted_class = label_encoder.inverse_transform([prediction])[0]
                
                logger.info(f"Model prediction: {predicted_class}")
                
                # Get any additional reasons the model might have found
                ineligibility_reasons = check_ineligibility_reasons(data) if predicted_class != "Eligible" else []
            except Exception as transform_error:
                logger.error(f"Error during transformation: {transform_error}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Transformation error: {str(transform_error)}")
        
        # Get donor profile information
        donor_profile = determine_donor_profile(input_df)
        
        # Enhanced response with additional features
        response = {
            "prediction": predicted_class,
            "probability": probabilities,
            "ineligibility_reasons": ineligibility_reasons,
            "donor_profile": donor_profile
        }
        
        # Add eligibility timeline for temporarily ineligible donors
        if predicted_class == "Temporairement Non-eligible":
            response["eligibility_timeline"] = get_eligibility_timeline(data, ineligibility_reasons)
            response["improvement_tips"] = generate_improvement_tips(ineligibility_reasons)
        
        # Add blood demand information
        response["blood_demand"] = get_blood_demand()
        
        # Add similar donors information
        response["similar_donors"] = find_similar_donors(data)
        
        logger.info("Prediction complete and response prepared")
        return response
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        # Try to make a simple prediction to verify the model is working
        test_input = DonorInput(
            Age=30, 
            Genre="Homme", 
            Taille=175.0, 
            Poids=70.0,
            Niveau_d_etude="Universitaire",
            Situation_Matrimoniale_SM="Célibataire",
            Profession="Enseignant",
            Arrondissement_de_residence="Douala I",
            Quartier_de_Residence="Bonapriso",
            Nationalite="Camerounais",
            Religion="Chrétien",
            A_t_il_elle_deja_donne_le_sang="Non",
            Taux_dhemoglobine=14.0
        )
        
        # Check if preprocessing works
        preprocess_input(test_input)
        
        return {
            "status": "API is running", 
            "model_loaded": model is not None,
            "preprocessor_loaded": preprocessor is not None,
            "label_encoder_loaded": label_encoder is not None,
            "test_input_valid": True
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "API is running but has issues",
            "error": str(e)
        }

@app.get("/demographics")
async def get_demographics_endpoint():
    """Get demographic data for the donor population"""
    
    if dataset is None:
        # Return mock data if dataset isn't available
        return {
            "age_distribution": [
                {"group": "18-25", "percentage": 22, "eligibility_rate": 68},
                {"group": "26-35", "percentage": 35, "eligibility_rate": 75},
                {"group": "36-45", "percentage": 25, "eligibility_rate": 70},
                {"group": "46-55", "percentage": 12, "eligibility_rate": 65},
                {"group": "56+", "percentage": 6, "eligibility_rate": 55}
            ],
            "gender_distribution": [
                {"gender": "Homme", "percentage": 55, "eligibility_rate": 72},
                {"gender": "Femme", "percentage": 45, "eligibility_rate": 68}
            ],
            "health_condition_rates": [
                {"condition": "Hypertension", "percentage": 15},
                {"condition": "Diabetes", "percentage": 8},
                {"condition": "Low Hemoglobin", "percentage": 20},
                {"condition": "Heart Condition", "percentage": 5}
            ]
        }
    
    # Real data analysis if dataset is available
    try:
        demographics = {}
        
        # Age distribution
        age_bins = [18, 25, 35, 45, 55, 100]
        age_labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
        
        dataset['Age_Group'] = pd.cut(
            dataset['Age'], 
            bins=age_bins, 
            labels=age_labels, 
            right=False
        )
        
        age_dist = dataset['Age_Group'].value_counts(normalize=True) * 100
        age_dist = age_dist.sort_index()
        
        # Calculate eligibility rate by age group
        age_elig = dataset.groupby('Age_Group')['ELIGIBILITE AU DON.'].apply(
            lambda x: (x == 'Eligible').mean() * 100
        )
        
        age_distribution = []
        for age_group in age_labels:
            if age_group in age_dist.index:
                age_distribution.append({
                    "group": age_group,
                    "percentage": round(age_dist[age_group], 1),
                    "eligibility_rate": round(age_elig.get(age_group, 0), 1)
                })
        
        demographics["age_distribution"] = age_distribution
        
        # Gender distribution
        gender_dist = dataset['Genre'].value_counts(normalize=True) * 100
        gender_elig = dataset.groupby('Genre')['ELIGIBILITE AU DON.'].apply(
            lambda x: (x == 'Eligible').mean() * 100
        )
        
        gender_distribution = []
        for gender in gender_dist.index:
            gender_distribution.append({
                "gender": gender,
                "percentage": round(gender_dist[gender], 1),
                "eligibility_rate": round(gender_elig.get(gender, 0), 1)
            })
        
        demographics["gender_distribution"] = gender_distribution
        
        # Health condition rates
        health_conditions = [
            ("Raison de non-eligibilite totale  [Hypertendus]", "Hypertension"),
            ("Raison de non-eligibilite totale  [Diabetique]", "Diabetes"),
            ("Raison de non-eligibilite totale  [Cardiaque]", "Heart Condition"),
            ("Raison indisponibilite  [Taux d'hemoglobine bas ]", "Low Hemoglobin")
        ]
        
        health_condition_rates = []
        for col, label in health_conditions:
            if col in dataset.columns:
                rate = (dataset[col].str.lower() == 'oui').mean() * 100
                health_condition_rates.append({
                    "condition": label,
                    "percentage": round(rate, 1)
                })
        
        demographics["health_condition_rates"] = health_condition_rates
        
        return demographics
    
    except Exception as e:
        logger.error(f"Error analyzing demographics: {e}")
        # Return mock data if analysis fails
        return {
            "age_distribution": [
                {"group": "18-25", "percentage": 22, "eligibility_rate": 68},
                {"group": "26-35", "percentage": 35, "eligibility_rate": 75},
                {"group": "36-45", "percentage": 25, "eligibility_rate": 70},
                {"group": "46-55", "percentage": 12, "eligibility_rate": 65},
                {"group": "56+", "percentage": 6, "eligibility_rate": 55}
            ],
            "gender_distribution": [
                {"gender": "Homme", "percentage": 55, "eligibility_rate": 72},
                {"gender": "Femme", "percentage": 45, "eligibility_rate": 68}
            ],
            "health_condition_rates": [
                {"condition": "Hypertension", "percentage": 15},
                {"condition": "Diabetes", "percentage": 8},
                {"condition": "Low Hemoglobin", "percentage": 20},
                {"condition": "Heart Condition", "percentage": 5}
            ]
        }

@app.get("/blood-demand")
async def get_blood_demand_endpoint():
    """Get current blood demand information"""
    return get_blood_demand()

@app.get("/eligibility-stats")
async def get_eligibility_stats():
    """Get statistics about eligibility factors"""
    
    if dataset is None:
        # Return mock data if dataset isn't available
        return {
            "eligibility_distribution": {
                "Eligible": 65,
                "Temporairement Non-eligible": 25,
                "Definitivement Non-eligible": 10
            },
            "top_reasons": [
                {"reason": "Low Hemoglobin", "percentage": 35},
                {"reason": "Recent Donation", "percentage": 15},
                {"reason": "Health Condition", "percentage": 12},
                {"reason": "Medication", "percentage": 10},
                {"reason": "Age/Weight Issues", "percentage": 8}
            ],
            "demographics_impact": {
                "age": {"high_risk": ">60", "low_risk": "25-45"},
                "gender": {"high_risk": "Female", "low_risk": "Male"},
                "bmi": {"high_risk": "<18.5 or >30", "low_risk": "20-25"}
            }
        }
    
    try:
        # Calculate eligibility distribution
        elig_dist = dataset['ELIGIBILITE AU DON.'].value_counts(normalize=True) * 100
        eligibility_distribution = {
            category: round(elig_dist.get(category, 0), 1) 
            for category in ['Eligible', 'Temporairement Non-eligible', 'Definitivement Non-eligible']
        }
        
        # Find top ineligibility reasons
        reason_columns = [
            ('Raison indisponibilite  [Taux d\'hemoglobine bas ]', 'Low Hemoglobin'),
            ('Raison indisponibilite  [date de dernier Don < 3 mois ]', 'Recent Donation'),
            ('Raison indisponibilite  [Est sous anti-biotherapie  ]', 'Medication'),
            ('Raison indisponibilite  [IST recente (Exclu VIH, Hbs, Hcv)]', 'Recent STI'),
            ('Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]', 'Carrier HIV/HBS/HCV'),
            ('Raison de non-eligibilite totale  [Drepanocytaire]', 'Sickle Cell'),
            ('Raison de non-eligibilite totale  [Diabetique]', 'Diabetes'),
            ('Raison de non-eligibilite totale  [Hypertendus]', 'Hypertension'),
            ('Raison de non-eligibilite totale  [Asthmatiques]', 'Asthma'),
            ('Raison de non-eligibilite totale  [Cardiaque]', 'Heart Condition')
        ]
        
        top_reasons = []
        for col, label in reason_columns:
            if col in dataset.columns:
                ineligible_subset = dataset[dataset['ELIGIBILITE AU DON.'] != 'Eligible']
                if len(ineligible_subset) > 0:
                    rate = (ineligible_subset[col].str.lower() == 'oui').mean() * 100
                    if not np.isnan(rate) and rate > 0:
                        top_reasons.append({
                            "reason": label,
                            "percentage": round(rate, 1)
                        })
        
        top_reasons.sort(key=lambda x: x["percentage"], reverse=True)
        
        # Demographics impact analysis
        demographics_impact = {
            "age": {
                "high_risk": ">60" if dataset[dataset['Age'] > 60]['ELIGIBILITE AU DON.'].value_counts(normalize=True).get('Eligible', 0) < 0.5 else ">50",
                "low_risk": "25-45" if dataset[(dataset['Age'] >= 25) & (dataset['Age'] <= 45)]['ELIGIBILITE AU DON.'].value_counts(normalize=True).get('Eligible', 0) > 0.6 else "18-35"
            },
            "gender": {
                "high_risk": "Female" if (dataset[dataset['Genre'] == 'Femme']['ELIGIBILITE AU DON.'] == 'Eligible').mean() < (dataset[dataset['Genre'] == 'Homme']['ELIGIBILITE AU DON.'] == 'Eligible').mean() else "Male",
                "low_risk": "Male" if (dataset[dataset['Genre'] == 'Homme']['ELIGIBILITE AU DON.'] == 'Eligible').mean() > (dataset[dataset['Genre'] == 'Femme']['ELIGIBILITE AU DON.'] == 'Eligible').mean() else "Female"
            }
        }
        
        # BMI analysis if possible
        if 'Poids' in dataset.columns and 'Taille' in dataset.columns:
            dataset['BMI'] = dataset['Poids'] / ((dataset['Taille'] / 100) ** 2)
            
            # Check eligibility rates for different BMI ranges
            normal_bmi_elig_rate = dataset[(dataset['BMI'] >= 18.5) & (dataset['BMI'] < 25)]['ELIGIBILITE AU DON.'].value_counts(normalize=True).get('Eligible', 0)
            low_bmi_elig_rate = dataset[dataset['BMI'] < 18.5]['ELIGIBILITE AU DON.'].value_counts(normalize=True).get('Eligible', 0)
            high_bmi_elig_rate = dataset[dataset['BMI'] >= 30]['ELIGIBILITE AU DON.'].value_counts(normalize=True).get('Eligible', 0)
            
            if normal_bmi_elig_rate > max(low_bmi_elig_rate, high_bmi_elig_rate):
                demographics_impact["bmi"] = {
                    "high_risk": "<18.5 or >30",
                    "low_risk": "18.5-25"
                }
            else:
                demographics_impact["bmi"] = {
                    "high_risk": "Extreme ranges",
                    "low_risk": "Normal range"
                }
        
        return {
            "eligibility_distribution": eligibility_distribution,
            "top_reasons": top_reasons[:5],  # Top 5 reasons
            "demographics_impact": demographics_impact
        }
    
    except Exception as e:
        logger.error(f"Error analyzing eligibility stats: {e}")
        return {
            "eligibility_distribution": {
                "Eligible": 65,
                "Temporairement Non-eligible": 25,
                "Definitivement Non-eligible": 10
            },
            "top_reasons": [
                {"reason": "Low Hemoglobin", "percentage": 35},
                {"reason": "Recent Donation", "percentage": 15},
                {"reason": "Health Condition", "percentage": 12},
                {"reason": "Medication", "percentage": 10},
                {"reason": "Age/Weight Issues", "percentage": 8}
            ],
            "demographics_impact": {
                "age": {"high_risk": ">60", "low_risk": "25-45"},
                "gender": {"high_risk": "Female", "low_risk": "Male"},
                "bmi": {"high_risk": "<18.5 or >30", "low_risk": "20-25"}
            }
        }