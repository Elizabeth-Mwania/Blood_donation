import streamlit as st
import requests
import pandas as pd
import re
from datetime import datetime

# Set page configuration
#st.set_page_config(page_title="Blood Donation Dashboard", layout="wide")

# UI Component Functions
def render_header():
    col1,col2 = st.columns(2)
    with col1:
        st.image("Images/blood2.png", width= 200)
    with col2:
        st.markdown("""
            <style>
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: linear-gradient(to right, #fff1f2, #fee2e2);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .logo-section {
                display: flex;
                align-items: center;
            }
            .title {
                color: #991b1b;
                font-size: 2.5em;
                font-weight: bold;
            }
            .subtitle {
                color: #dc2626;
                font-size: 1.2em;
            }
            .tagline {
                text-align: right;
                color: #7f1d1d;
            }
            </style>
            <div class="header">
                <div class="logo-section">
                    <div>
                        <div class="subtitle">Blood Donation Platform</div>
                    </div>
                </div>
                <div class="tagline">
                    <div style="font-size: 1.2em; color: #dc2626;">
                        ❤️ Every Drop Saves Lives
                    </div>
                    <div style="font-size: 1em; color: #7f1d1d;">
                        Connecting Donors | Saving Communities
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_styles():
    st.markdown("""
        <style>
            .main-header {font-size: 2.5rem; color: #B22222; text-align: center; margin-bottom: 1rem;}
            .sub-header {font-size: 1.8rem; color: #8B0000; margin-top: 1rem;}
            .metric-container {background-color: #F8F8F8; border-radius: 5px; padding: 1rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
            .metric-container2 {background-color: red; border-radius: 5px; padding: 1rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
            .highlight {color: black; font-weight: bold;}
            .chart-container {background-color: white; border-radius: 5px; padding: 1rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin: 1rem 0;}
            .footer {text-align: center; font-size: 0.8rem; color: gray; margin-top: 2rem;}
            body {
                background: linear-gradient(-45deg, darkred, red, orangered, tomato);
                background-size: 400% 400%;
                animation: gradient 15s ease infinite;
            }
            @keyframes gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
        </style>
        <div class="main-header">🩸 Blood Donation Campaign Dashboard</div>
    """, unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .important-field { background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #1e90ff; }
    .important-label { color: #1e90ff; font-weight: bold; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
    .stButton>button:hover { background-color: #45a049; }
    </style>
""", unsafe_allow_html=True)

st.title("Blood Donation Campaign Dashboard")

# Load dataset for dropdowns
try:
    data = pd.read_csv("data/data_2019_cleaned.csv")
    if data.empty:
        raise ValueError("Dataset 'data_2019_cleaned.csv' is empty.")
    professions = sorted(data["Profession"].str.lower().unique().tolist())
    districts = sorted(data["Arrondissement de residence"].str.lower().unique().tolist())
    neighborhoods = sorted(data["Quartier de Residence"].str.lower().unique().tolist())
    health_condition_cols = [
        'Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]', 'Raison de non-eligibilite totale  [Drepanocytaire]',
        'Raison de non-eligibilite totale  [Diabetique]', 'Raison de non-eligibilite totale  [Hypertendus]',
        'Raison de non-eligibilite totale  [Asthmatiques]', 'Raison de non-eligibilite totale  [Cardiaque]',
        'Raison de non-eligibilite totale  [Tatoue]', 'Raison de non-eligibilite totale  [Scarifie]'
    ]
    health_condition_mapping = {
        'Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]': 'Porteur_HIV_hbs_hcv',
        'Raison de non-eligibilite totale  [Drepanocytaire]': 'Drepanocytaire',
        'Raison de non-eligibilite totale  [Diabetique]': 'Diabetique',
        'Raison de non-eligibilite totale  [Hypertendus]': 'Hypertendus',
        'Raison de non-eligibilite totale  [Asthmatiques]': 'Asthmatiques',
        'Raison de non-eligibilite totale  [Cardiaque]': 'Cardiaque',
        'Raison de non-eligibilite totale  [Tatoue]': 'Tatoue',
        'Raison de non-eligibilite totale  [Scarifie]': 'Scarifie'
    }
    health_conditions_display = [col.split('[')[1].split(']')[0] for col in health_condition_cols]
except FileNotFoundError:
    st.warning("Dataset 'data_2019_cleaned.csv' not found. Using default options.")
    professions = ["enseignant", "étudiant", "commerçant"]
    districts = ["douala i", "douala ii"]
    neighborhoods = ["bonapriso", "akwa"]
    health_condition_cols = [
        'Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]', 'Raison de non-eligibilite totale  [Drepanocytaire]',
        'Raison de non-eligibilite totale  [Diabetique]', 'Raison de non-eligibilite totale  [Hypertendus]',
        'Raison de non-eligibilite totale  [Asthmatiques]', 'Raison de non-eligibilite totale  [Cardiaque]',
        'Raison de non-eligibilite totale  [Tatoue]', 'Raison de non-eligibilite totale  [Scarifie]'
    ]
    health_condition_mapping = {
        'Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]': 'Porteur_HIV_hbs_hcv',
        'Raison de non-eligibilite totale  [Drepanocytaire]': 'Drepanocytaire',
        'Raison de non-eligibilite totale  [Diabetique]': 'Diabetique',
        'Raison de non-eligibilite totale  [Hypertendus]': 'Hypertendus',
        'Raison de non-eligibilite totale  [Asthmatiques]': 'Asthmatiques',
        'Raison de non-eligibilite totale  [Cardiaque]': 'Cardiaque',
        'Raison de non-eligibilite totale  [Tatoue]': 'Tatoue',
        'Raison de non-eligibilite totale  [Scarifie]': 'Scarifie'
    }
    health_conditions_display = [col.split('[')[1].split(']')[0] for col in health_condition_cols]
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}. Using default options.")
    professions = ["enseignant", "étudiant", "commerçant"]
    districts = ["douala i", "douala ii"]
    neighborhoods = ["bonapriso", "akwa"]
    health_condition_cols = [
        'Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]', 'Raison de non-eligibilite totale  [Drepanocytaire]',
        'Raison de non-eligibilite totale  [Diabetique]', 'Raison de non-eligibilite totale  [Hypertendus]',
        'Raison de non-eligibilite totale  [Asthmatiques]', 'Raison de non-eligibilite totale  [Cardiaque]',
        'Raison de non-eligibilite totale  [Tatoue]', 'Raison de non-eligibilite totale  [Scarifie]'
    ]
    health_condition_mapping = {
        'Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]': 'Porteur_HIV_hbs_hcv',
        'Raison de non-eligibilite totale  [Drepanocytaire]': 'Drepanocytaire',
        'Raison de non-eligibilite totale  [Diabetique]': 'Diabetique',
        'Raison de non-eligibilite totale  [Hypertendus]': 'Hypertendus',
        'Raison de non-eligibilite totale  [Asthmatiques]': 'Asthmatiques',
        'Raison de non-eligibilite totale  [Cardiaque]': 'Cardiaque',
        'Raison de non-eligibilite totale  [Tatoue]': 'Tatoue',
        'Raison de non-eligibilite totale  [Scarifie]': 'Scarifie'
    }
    health_conditions_display = [col.split('[')[1].split(']')[0] for col in health_condition_cols]

# Tabs
tab1, tab2 = st.tabs(["Eligibility Prediction", "Health Conditions Analysis"])

# Eligibility Prediction Tab
with tab1:
    st.header("Predict Donor Eligibility")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographic Information")
        st.markdown('<p class="important-label">Age *</p>', unsafe_allow_html=True)
        age = st.number_input("Enter your age", min_value=18, max_value=100, value=30)
        if age < 18 or age > 100:
            st.error("Age must be between 18 and 100.")
        
        gender = st.selectbox("Gender", ["Homme", "Femme"], index=0)
        
        st.markdown('<p class="important-label">Height (cm) *</p>', unsafe_allow_html=True)
        height = st.number_input("Enter your height in cm", min_value=100, max_value=250, value=170)
        if height < 100 or height > 250:
            st.error("Height must be between 100 and 250 cm.")
        
        st.markdown('<p class="important-label">Weight (kg) *</p>', unsafe_allow_html=True)
        weight = st.number_input("Enter your weight in kg", min_value=30, max_value=200, value=70)
        if weight < 50:
            st.warning("Weight below 50 kg may make you ineligible.")
        
        if height and weight:
            bmi = weight / (height / 100) ** 2
            st.markdown(f"*Your BMI:* {bmi:.2f}")
            if bmi < 18.5:
                st.warning("BMI below 18.5 may affect eligibility.")
        
        education = st.selectbox("Education Level", ["Primaire", "Secondaire", "Universitaire"], index=2)
        marital = st.selectbox("Marital Status", ["Célibataire", "Marié"], index=0)
        st.subheader("Profession")
        new_profession = st.text_input("Add a new profession (lowercase, no special characters)", "")
        if new_profession:
            new_profession = new_profession.lower()
            if re.match("^[a-z]+$", new_profession) and new_profession not in professions:
                professions.append(new_profession)
                st.success(f"Added profession: {new_profession}")
            elif not re.match("^[a-z]+$", new_profession):
                st.error("Profession must contain only lowercase letters (a-z).")
        profession = st.selectbox("Profession", professions, index=0)
        st.subheader("Location")
        district = st.selectbox("District of Residence", districts, index=0)
        neighborhood = st.selectbox("Neighborhood of Residence", neighborhoods, index=0)
    
    with col2:
        st.subheader("Donation and Health Information")
        nationality = st.selectbox("Nationality", ["Camerounais"], index=0)
        religion = st.selectbox("Religion", ["Chrétien", "Musulman"], index=0)
        
        donated = st.selectbox("Has Donated Before?", ["Oui", "Non"], index=0)
        last_donation = "2000-01-01" if donated == "Non" else st.text_input("Date of Last Donation (YYYY-MM-DD)", "", help="Required if Oui")
        if donated == "Oui" and last_donation:
            try:
                datetime.strptime(last_donation, "%Y-%m-%d")
            except ValueError:
                st.error("Date must be in YYYY-MM-DD format.")
        
        st.markdown('<p class="important-label">Hemoglobin Level (g/dL) *</p>', unsafe_allow_html=True)
        hemoglobin = st.number_input("Hemoglobin level", min_value=5.0, max_value=20.0, value=14.5)
        if hemoglobin < 12.5 and gender == "Femme":
            st.warning("Hemoglobin below 12.5 g/dL may make you ineligible (women).")
        elif hemoglobin < 13.0 and gender == "Homme":
            st.warning("Hemoglobin below 13.0 g/dL may make you ineligible (men).")
        
        # Female-specific conditions
        female_conditions = {
            "La_DDR_est_mauvais_si_14_jour_avant_le_don": "Non",
            "Allaitement": "Non",
            "A_accoucher_ces_6_derniers_mois": "Non",
            "Interruption_de_grossesse_ces_06_derniers_mois": "Non"
        }
        if gender == "Femme":
            st.subheader("Female-Specific Information")
            ddr_date = st.text_input("Date of Last Menstrual Period (YYYY-MM-DD)", "", help="Format: YYYY-MM-DD")
            if ddr_date:
                try:
                    ddr_datetime = datetime.strptime(ddr_date, "%Y-%m-%d")
                    days_since_ddr = (datetime.now() - ddr_datetime).days
                    female_conditions["La_DDR_est_mauvais_si_14_jour_avant_le_don"] = "Oui" if days_since_ddr < 14 else "Non"
                except ValueError:
                    st.error("DDR date must be in YYYY-MM-DD format.")
            female_conditions["Allaitement"] = st.selectbox("Currently Breastfeeding?", ["Oui", "Non"], index=1)
            female_conditions["A_accoucher_ces_6_derniers_mois"] = st.selectbox("Delivered in Last 6 Months?", ["Oui", "Non"], index=1)
            female_conditions["Interruption_de_grossesse_ces_06_derniers_mois"] = st.selectbox("Pregnancy Termination in Last 6 Months?", ["Oui", "Non"], index=1)
        
        st.markdown("### Health Conditions")
        has_conditions = st.checkbox("Do you have any health conditions?")
        health_conditions = {v: "Non" for v in health_condition_mapping.values()}
        if has_conditions:
            selected_conditions = st.multiselect("Select health conditions", options=health_conditions_display)
            for condition in selected_conditions:
                dataset_col = next(col for col, display in zip(health_condition_cols, health_conditions_display) if display == condition)
                health_conditions[health_condition_mapping[dataset_col]] = "Oui"
    
    if st.button("Predict"):
        if donated == "Oui" and not last_donation:
            st.error("Please provide the date of your last donation.")
            st.stop()
        
        input_data = {
            "Age": age,
            "Genre": gender,
            "Taille": height,
            "Poids": weight,
            "Niveau_d_etude": education,
            "Situation_Matrimoniale_SM": marital,
            "Profession": profession,
            "Arrondissement_de_residence": district,
            "Quartier_de_Residence": neighborhood,
            "Nationalite": nationality,
            "Religion": religion,
            "A_t_il_elle_deja_donne_le_sang": donated,
            "Si_oui_preciser_la_date_du_dernier_don": last_donation,
            "Taux_dhemoglobine": hemoglobin,
            "Porteur_HIV_hbs_hcv": health_conditions["Porteur_HIV_hbs_hcv"],
            "Drepanocytaire": health_conditions["Drepanocytaire"],
            "Diabetique": health_conditions["Diabetique"],
            "Hypertendus": health_conditions["Hypertendus"],
            "Asthmatiques": health_conditions["Asthmatiques"],
            "Cardiaque": health_conditions["Cardiaque"],
            "Tatoue": health_conditions["Tatoue"],
            "Scarifie": health_conditions["Scarifie"],
            "La_DDR_est_mauvais_si_14_jour_avant_le_don": female_conditions["La_DDR_est_mauvais_si_14_jour_avant_le_don"],
            "Allaitement": female_conditions["Allaitement"],
            "A_accoucher_ces_6_derniers_mois": female_conditions["A_accoucher_ces_6_derniers_mois"],
            "Interruption_de_grossesse_ces_06_derniers_mois": female_conditions["Interruption_de_grossesse_ces_06_derniers_mois"]
        }

        api_url = "http://127.0.0.1:8000"
        try:
            health_response = requests.get(f"{api_url}/health", timeout=2)
            if health_response.status_code != 200:
                st.error("API is not running. Start with 'uvicorn api:app --reload'.")
                st.stop()
        except requests.exceptions.RequestException:
            st.error("API is not running. Start with 'uvicorn api:app --reload'.")
            st.stop()

        with st.spinner("Making prediction..."):
            try:
                response = requests.post(f"{api_url}/predict", json=input_data, timeout=5)
                response.raise_for_status()
                result = response.json()
                prediction = result["prediction"]
                probabilities = result["probability"]
                st.success(f"Prediction: {prediction}")
                st.write(f"Probabilities: Definitivement non-eligible: {probabilities[0]:.2f}, "
                         f"Eligible: {probabilities[1]:.2f}, Temporairement Non-eligible: {probabilities[2]:.2f}")
                
                # Display ineligibility reasons if not eligible
                if prediction != "Eligible":
                    st.subheader("Reasons for Ineligibility")
                    reasons = result.get("ineligibility_reasons", [])
                    if reasons:
                        for r in reasons:
                            severity = r["severity"]
                            color = "red" if severity >= 4 else "orange" if severity == 3 else "yellow"
                            st.markdown(
                                f"<span style='color: {color}; font-weight: bold;'>[Severity {severity}/5 - {r['type']}]</span> {r['reason']}",
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("No specific reasons identified; ineligibility based on combined factors.")
            except requests.exceptions.HTTPError as http_err:
                st.error(f"API error: {http_err.response.status_code} - {http_err.response.text}")
            except requests.exceptions.RequestException as req_err:
                st.error(f"Error connecting to API: {str(req_err)}")

# Health Conditions Analysis Tab
with tab2:
    st.header("Health Conditions Among Donors")
    try:
        data = pd.read_csv("data_2019_cleaned.csv")
        health_stats = {}
        for col in health_condition_cols:
            count = data[col].str.lower().map({'oui': 1, 'non': 0}).sum()
            health_stats[col.split('[')[1].split(']')[0]] = (count / len(data)) * 100
        st.bar_chart(health_stats)
    except FileNotFoundError:
        st.error("Dataset 'data_2019_cleaned.csv' not found.")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")