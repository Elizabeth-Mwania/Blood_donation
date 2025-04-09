import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import json
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("blood_donation_dashboard")

# Set page configuration
#st.set_page_config(page_title="Blood Donation Dashboard", layout="wide")

# Custom CSS with improved styling
st.markdown("""
    <style>
    .important-field { background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #1e90ff; }
    .important-label { color: #1e90ff; font-weight: bold; }
    .success-header { color: #4CAF50; font-weight: bold; }
    .warning-header { color: #FFA500; font-weight: bold; }
    .error-header { color: #FF0000; font-weight: bold; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
    .stButton>button:hover { background-color: #45a049; }
    .severity-5 { color: #d32f2f; font-weight: bold; }
    .severity-4 { color: #f44336; font-weight: bold; }
    .severity-3 { color: #ff9800; font-weight: bold; }
    .severity-2 { color: #ffc107; font-weight: bold; }
    .severity-1 { color: #8bc34a; font-weight: bold; }
    .reason-card { 
        background-color: #f9f9f9; 
        border-left: 4px solid #ccc; 
        padding: 10px; 
        margin: 5px 0; 
        border-radius: 0 5px 5px 0;
    }
    .reason-permanent { border-left-color: #d32f2f; }
    .reason-temporary { border-left-color: #ff9800; }
    .reason-warning { border-left-color: #8bc34a; }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🩺 Lab : Eligibility checking !")
st.markdown("""
This application predicts blood donation eligibility using a machine learning model. 
Enter your details to check your eligibility status and understand the reasons for ineligibility, if any.
""")

# Initialize session state
if 'gender' not in st.session_state:
    st.session_state.gender = "Homme"
if 'show_health_conditions' not in st.session_state:
    st.session_state.show_health_conditions = False
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'api_response' not in st.session_state:
    st.session_state.api_response = None
if 'api_error' not in st.session_state:
    st.session_state.api_error = None

# Load dataset for dropdowns
try:
    data = pd.read_csv("data/data_2019_cleaned.csv")
    if data.empty:
        raise ValueError("Dataset 'data_2019_cleaned.csv' is empty.")
    
    # Debug: Show available columns
    logger.info("Available columns in dataset: %s", data.columns.tolist())
    
    # Add hemoglobin field if needed
    if "Taux d'hemoglobine" in data.columns and "Taux_dhemoglobine" not in data.columns:
        data["Taux_dhemoglobine"] = data["Taux d'hemoglobine"]
    
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

# Helper function to check API status with retry mechanism
def check_api_status(api_url, max_retries=3, retry_delay=1):
    for attempt in range(max_retries):
        try:
            health_response = requests.get(f"{api_url}/health", timeout=2)
            if health_response.status_code == 200:
                return True, "API is running"
            else:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return False, f"API error: {health_response.status_code}"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return False, f"API is not running. Error: {str(e)}"
# Function to make API prediction with error handling
def predict_eligibility(api_url, input_data):
    try:
        with st.spinner("Making prediction..."):
            # Log the data being sent (debugging)
            logger.info(f"Sending data to API: {json.dumps(input_data)}")
            
            # Make the API request with increased timeout
            response = requests.post(f"{api_url}/predict", json=input_data, timeout=15)
            
            # Log the response status
            logger.info(f"API Response Status Code: {response.status_code}")
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse and return the result
            result = response.json()
            return True, result
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return False, "The API request timed out. The server might be overloaded or temporarily unavailable."
    except requests.exceptions.ConnectionError:
        logger.error("Connection error when calling API")
        return False, "Connection error. The API server might be down or unreachable."
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error: {http_err}")
        try:
            error_details = http_err.response.json()
            return False, f"HTTP error {http_err.response.status_code}: {error_details.get('detail', '')}"
        except:
            return False, f"HTTP error {http_err.response.status_code}: {http_err.response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return False, f"Request error: {str(e)}"
    except ValueError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return False, "Invalid response from API (not valid JSON)"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False, f"Unexpected error: {str(e)}"

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Eligibility Prediction", "Health Conditions Analysis", "Dataset Exploration"])

# Tab 1: Eligibility Prediction
with tab1:
    st.header("Predict Donor Eligibility")
    st.markdown("Fill in the details below to predict your eligibility for blood donation.")

    # API status indicator
    api_url = "http://127.0.0.1:8000"
    api_status, api_message = check_api_status(api_url)
    
    if api_status:
        st.success(api_message)
    else:
        st.error(f"{api_message} Please start the API server before using this application.")
        st.info("If you're encountering issues, try the following troubleshooting steps:")
        st.code("1. Make sure you're running the API server with: uvicorn api:app --reload")
        st.code("2. Check that you have all required models in the 'models/' directory")
        st.code("3. Verify that the dataset 'data_2019_cleaned.csv' is accessible")
        st.stop()

    # Gender selection
    gender = st.radio("Select Gender", ["Homme", "Femme"], horizontal=True)
    st.session_state.gender = gender
    
    # Health conditions visibility toggle
    has_conditions = st.checkbox("Do you have any health conditions?", value=st.session_state.show_health_conditions)
    st.session_state.show_health_conditions = has_conditions

    # Input form
    with st.form("donor_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Demographic Information")
            st.markdown('<p class="important-label">Age *</p>', unsafe_allow_html=True)
            age = st.number_input("Enter your age", min_value=16, max_value=100, value=30)
            if age < 18 or age > 65:
                st.warning("Age outside 18-65 may affect eligibility.")

            st.markdown('<p class="important-label">Height (cm) *</p>', unsafe_allow_html=True)
            height = st.number_input("Enter your height in cm", min_value=100.0, max_value=250.0, value=170.0)
            if height < 100 or height > 250:
                st.error("Height must be between 100 and 250 cm.")

            st.markdown('<p class="important-label">Weight (kg) *</p>', unsafe_allow_html=True)
            weight = st.number_input("Enter your weight in kg", min_value=30.0, max_value=200.0, value=70.0)
            if weight < 50:
                st.warning("Weight below 50 kg may make you ineligible.")

            if height and weight:
                bmi = weight / (height / 100) ** 2
                if bmi < 18.5 or bmi >= 30:
                    st.warning(f"BMI: {bmi:.2f} - Outside healthy range (18.5-30) may affect eligibility.")
                else:
                    st.info(f"BMI: {bmi:.2f}")

            education = st.selectbox("Education Level", ["Primaire", "Secondaire", "Universitaire", "Aucun"])
            marital = st.selectbox("Marital Status", ["Célibataire", "Marié", "Divorcé", "Veuf"])
            st.subheader("Profession")
            new_profession = st.text_input("Add a new profession (if not in list)", "")
            profession = st.selectbox("Profession", professions + ([new_profession] if new_profession else []), index=0)
            st.subheader("Location")
            district = st.selectbox("District of Residence", districts)
            neighborhood = st.selectbox("Neighborhood of Residence", neighborhoods)

        with col2:
            st.subheader("Donation and Health Information")
            nationality = st.selectbox("Nationality", ["Camerounais", "Français", "Autre"])
            religion = st.selectbox("Religion", ["Chrétien", "Musulman", "Autre"])
            
            donated = st.selectbox("Have you donated blood before?", ["Non", "Oui"])
            last_donation_date = None
            if donated == "Oui":
                last_donation_date = st.date_input(
                    "Date of Last Donation",
                    value=None,
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime.now()
                )
                if last_donation_date:
                    days_since = (datetime.now().date() - last_donation_date).days
                    if days_since < 90:
                        st.warning(f"Last donation was {days_since} days ago. Minimum 90 days required.")

            st.markdown('<p class="important-label">Hemoglobin Level (g/dL) *</p>', unsafe_allow_html=True)
            hemoglobin = st.number_input("Hemoglobin level", min_value=5.0, max_value=20.0, value=14.5, step=0.1)
            if st.session_state.gender == "Femme" and hemoglobin < 12.5:
                st.warning(f"Hemoglobin {hemoglobin} g/dL below threshold for women (12.5 g/dL).")
            elif st.session_state.gender == "Homme" and hemoglobin < 13.0:
                st.warning(f"Hemoglobin {hemoglobin} g/dL below threshold for men (13.0 g/dL).")

            st.subheader("Medication")
            on_medication = st.selectbox("Are you currently on antibiotics or other medication?", ["Non", "Oui"])

            health_conditions = {v: "Non" for v in health_condition_mapping.values()}
            if st.session_state.show_health_conditions:
                st.subheader("Health Conditions")
                selected_conditions = st.multiselect("Select health conditions", options=health_conditions_display)
                for condition in selected_conditions:
                    dataset_col = next(col for col, display in zip(health_condition_cols, health_conditions_display) if display == condition)
                    health_conditions[health_condition_mapping[dataset_col]] = "Oui"

            had_surgery = st.selectbox("Have you had surgery in the last 6 months?", ["Non", "Oui"])
            had_transfusion = st.selectbox("Have you received a blood transfusion in the past?", ["Non", "Oui"])
            had_sti = st.selectbox("Have you had a sexually transmitted infection recently?", ["Non", "Oui"])

            female_conditions = {
                "La_DDR_est_mauvais_si_14_jour_avant_le_don": "Non",
                "Allaitement": "Non",
                "A_accoucher_ces_6_derniers_mois": "Non",
                "Interruption_de_grossesse_ces_06_derniers_mois": "Non",
                "Est_enceinte": "Non"
            }
            
            if st.session_state.gender == "Femme":
                st.subheader("Female-Specific Information")
                ddr_date = st.date_input(
                    "Date of Last Menstrual Period", 
                    value=None,
                    min_value=datetime(2023, 1, 1),
                    max_value=datetime.now()
                )
                if ddr_date:
                    days_since_ddr = (datetime.now().date() - ddr_date).days
                    female_conditions["La_DDR_est_mauvais_si_14_jour_avant_le_don"] = "Oui" if days_since_ddr < 14 else "Non"
                    if days_since_ddr < 14:
                        st.warning(f"Last period {days_since_ddr} days ago may affect eligibility.")
                
                female_conditions["Est_enceinte"] = st.selectbox("Are you currently pregnant?", ["Non", "Oui"])
                female_conditions["Allaitement"] = st.selectbox("Are you currently breastfeeding?", ["Non", "Oui"])
                female_conditions["A_accoucher_ces_6_derniers_mois"] = st.selectbox("Have you given birth in the last 6 months?", ["Non", "Oui"])
                female_conditions["Interruption_de_grossesse_ces_06_derniers_mois"] = st.selectbox("Have you had a pregnancy termination in the last 6 months?", ["Non", "Oui"])

        submitted = st.form_submit_button("Predict")
        if submitted:
            st.session_state.form_submitted = True

    # Prediction logic
    if st.session_state.form_submitted:
        if donated == "Oui" and not last_donation_date:
            st.error("Please provide the date of your last donation.")
            st.session_state.form_submitted = False
        else:
            # Prepare input data for API with correct field names
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
                "Si_oui_preciser_la_date_du_dernier_don": last_donation_date.strftime("%Y-%m-%d") if last_donation_date else "",
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
                "Interruption_de_grossesse_ces_06_derniers_mois": female_conditions["Interruption_de_grossesse_ces_06_derniers_mois"],
                "Est_sous_anti_biotherapie": on_medication,
                "Est_enceinte": female_conditions["Est_enceinte"],
                "Antecedent_de_transfusion": had_transfusion,
                "Recent_surgery": had_surgery,
                "Recent_STI": had_sti,
                "Si_autres_raison_preciser": "",
                "Autre_raison_preciser": ""
            }
            
            # Call the API and handle response
            success, result = predict_eligibility(api_url, input_data)
            
            if success:
                st.session_state.api_response = result
                st.session_state.api_error = None
            else:
                st.session_state.api_error = result
                st.session_state.api_response = None
            
            st.session_state.form_submitted = False

    # Display prediction results if available
    if st.session_state.api_error:
        st.error(f"API Error: {st.session_state.api_error}")
        st.warning("Please try again or check the API server status.")
    
    elif st.session_state.api_response:
        result = st.session_state.api_response
        prediction = result["prediction"]
        probabilities = result["probability"]
        
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            if prediction == "Eligible":
                st.markdown('<h2 class="success-header">✅ ELIGIBLE</h2>', unsafe_allow_html=True)
                st.balloons()
            elif prediction == "Temporairement Non-eligible":
                st.markdown('<h2 class="warning-header">⚠ TEMPORARILY INELIGIBLE</h2>', unsafe_allow_html=True)
            else:
                st.markdown('<h2 class="error-header">❌ PERMANENTLY INELIGIBLE</h2>', unsafe_allow_html=True)
            
            st.write("### Probability Breakdown")
            categories = ["Permanently Ineligible", "Eligible", "Temporarily Ineligible"]
            prob_df = pd.DataFrame({
                "Category": categories,
                "Probability": [round(p*100, 1) for p in probabilities]
            })
            
            # Create a Plotly bar chart
            colors = {'Eligible': '#4CAF50', 'Permanently Ineligible': '#d32f2f', 'Temporarily Ineligible': '#ff9800'}
            fig = px.bar(
                prob_df, 
                x="Probability", 
                y="Category", 
                orientation='h',
                text="Probability",
                color="Category",
                color_discrete_map=colors
            )
            fig.update_layout(
                title="Eligibility Probability Distribution",
                xaxis_title="Probability (%)",
                yaxis_title="",
                xaxis_range=[0, 100],
                height=300
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with result_col2:
            if prediction != "Eligible":
                st.write("### Reasons for Ineligibility")
                reasons = result.get("ineligibility_reasons", [])
                if reasons:
                    for r in reasons:
                        severity = r["severity"]
                        reason_type = r["type"]
                        severity_class = f"severity-{severity}"
                        type_class = f"reason-{reason_type.lower()}"
                        
                        explanation_mapping = {
                            "hemoglobin below 12.5": "Low hemoglobin may indicate anemia. Consult a doctor.",
                            "hemoglobin below 13.0": "Low hemoglobin may indicate anemia. Consult a doctor.",
                            "bmi": "Extreme BMI values may pose health risks during donation.",
                            "underweight": "Low BMI may affect your safety during donation.",
                            "obese": "High BMI requires additional health screening.",
                            "age": "Age outside 18-65 requires special consideration.",
                            "donated": "Wait at least 90 days between donations.",
                            "recent donation": "Wait at least 90 days between donations.",
                            "medication": "Some medications affect eligibility.",
                            "antibiotics": "Finish your antibiotics course before donating.",
                            "tattoo": "Wait 4 months after tattoos due to infection risk.",
                            "scarification": "Wait 4 months after scarification due to infection risk.",
                            "pregnant": "Pregnancy or recent childbirth affects eligibility.",
                            "breastfeeding": "Breastfeeding mothers should wait before donating.",
                            "menstruation": "Recent menstruation may lower iron levels.",
                            "hiv": "This condition permanently affects donation eligibility.",
                            "hepatitis": "This condition permanently affects donation eligibility.",
                            "diabetes": "This condition may permanently affect donation eligibility.",
                            "hypertension": "This condition may permanently affect donation eligibility.",
                            "heart": "Heart conditions may permanently affect eligibility.",
                            "asthma": "This condition may affect donation eligibility.",
                            "sickle cell": "This condition permanently affects donation eligibility.",
                            "surgery": "Recent surgery requires a waiting period.",
                            "transfusion": "Previous transfusions require a waiting period."
                        }
                        
                        # Find matching explanation
                        explanation = "This condition impacts eligibility."
                        for key, value in explanation_mapping.items():
                            if key.lower() in r["reason"].lower():
                                explanation = value
                                break
                        
                        st.markdown(
                            f"""
                            <div class="reason-card {type_class}">
                                <div class="{severity_class}">Severity {severity}/5 - {reason_type}</div>
                                <div style="font-size: 1.1em; margin: 5px 0;">{r["reason"]}</div>
                                <div style="font-size: 0.9em; color: #555;">{explanation}</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                    # Show eligibility timeline if available
                    if "eligibility_timeline" in result and result["eligibility_timeline"]:
                        st.write("### When You Can Donate")
                        timeline_data = result["eligibility_timeline"]
                        
                        # Create a Plotly timeline chart
                        timeline_df = pd.DataFrame(timeline_data)
                        timeline_df['days_to_wait'] = pd.to_numeric(timeline_df['days_to_wait'])
                        timeline_df = timeline_df.sort_values('days_to_wait')
                        
                        fig = px.timeline(
                            timeline_df,
                            x_start=0,
                            x_end='days_to_wait',
                            y='reason',
                            color='days_to_wait',
                            color_continuous_scale='Reds',
                            labels={'days_to_wait': 'Days to Wait', 'reason': 'Reason'}
                        )
                        fig.update_layout(
                            title="Waiting Period Before Eligibility",
                            xaxis_title="Days from Today",
                            yaxis_title="",
                            height=300,
                        )
                        
                        # Add current date and eligibility date markers
                        for i, row in timeline_df.iterrows():
                            fig.add_annotation(
                                x=row['days_to_wait'],
                                y=row['reason'],
                                text=row['eligible_after'],
                                showarrow=True,
                                arrowhead=1,
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show improvement tips if available
                    if "improvement_tips" in result and result["improvement_tips"]:
                        st.write("### Tips to Improve Eligibility")
                        tips = result["improvement_tips"]
                        for tip in tips:
                            st.markdown(
                                f"""
                                <div style="border-left: 4px solid #4CAF50; padding: 10px; margin: 5px 0; background-color: #f8f8f8;">
                                    <div style="font-weight: bold; color: #4CAF50;">{tip["category"]}: {tip["tip"]}</div>
                                    <div>{tip["description"]}</div>
                                    <div style="margin-top: 5px; font-size: 0.8em;">
                                        <span style="background-color: {'#f0f4c3' if tip['difficulty'] == 'Easy' else '#ffccbc' if tip['difficulty'] == 'Hard' else '#cfd8dc'}; padding: 2px 5px; border-radius: 3px;">
                                            {tip["difficulty"]}
                                        </span>
                                        <span style="margin-left: 10px;">{tip["time_to_implement"]}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No specific reasons identified.")
            else:
                st.write("### Eligibility Confirmed")
                st.markdown("""
                ✅ *You are eligible to donate blood!*
                
                *Next steps:*
                - Pre-donation screening (BP, temp, hemoglobin)
                - Donation (~450ml, 10-15 mins)
                - Rest and recover (10-15 mins)
                
                *Tips:*
                - Eat well and hydrate
                - Sleep well
                - Bring ID
                
                Thank you for saving lives! 🩸
                """)
                
                # Show blood demand information
                if "blood_demand" in result and result["blood_demand"]:
                    st.write("### Current Blood Demand")
                    blood_demand = result["blood_demand"]
                    
                    # Create a Plotly chart for blood demand
                    blood_df = pd.DataFrame(blood_demand)
                    
                    fig = px.bar(
                        blood_df,
                        x="blood_type",
                        y="urgency_level",
                        color="urgency_level",
                        color_continuous_scale="Reds",
                        labels={"blood_type": "Blood Type", "urgency_level": "Urgency Level"},
                        hover_data=["current_demand", "donation_impact"]
                    )
                    fig.update_layout(
                        title="Current Blood Type Demand",
                        xaxis_title="Blood Type",
                        yaxis_title="Urgency Level (1-5)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Show similar donors information if available
        if "similar_donors" in result and result["similar_donors"]:
            st.write("### Similar Donor Profiles")
            similar = result["similar_donors"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Similar Donors Found", similar["similar_count"])
                st.write(f"**Common Factors:** {', '.join(similar['common_factors'])}")
            
            with col2:
                # Create a gauge chart for eligibility rate
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=similar["eligibility_rate"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Eligibility Rate Among Similar Donors"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': similar["eligibility_rate"]
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
# Tab 2: Health Conditions Analysis
with tab2:
    st.header("Health Conditions Among Donors")
    st.markdown("Analyze the prevalence of health conditions affecting eligibility.")
    
    try:
        data = pd.read_csv("data/data_2019_cleaned.csv")
        
        # Create a column name mapping to handle potential column name variations
        hemoglobin_column = None
        if "Taux d'hemoglobine" in data.columns:
            hemoglobin_column = "Taux d'hemoglobine"
        elif "Taux_dhemoglobine" in data.columns:
            hemoglobin_column = "Taux_dhemoglobine"
        
        # Calculate health condition statistics
        health_stats = {col.split('[')[1].split(']')[0]: (data[col].str.lower().map({'oui': 1, 'non': 0}).sum() / len(data)) * 100 for col in health_condition_cols}
        
        st.subheader("Prevalence of Health Conditions")
        # Create Plotly horizontal bar chart
        health_df = pd.DataFrame({
            'Condition': list(health_stats.keys()),
            'Percentage': list(health_stats.values())
        }).sort_values('Percentage', ascending=False)
        
        fig = px.bar(
            health_df,
            x='Percentage',
            y='Condition',
            orientation='h',
            color='Percentage',
            color_continuous_scale='Reds',
            labels={'Percentage': 'Percentage of Donors (%)', 'Condition': 'Health Condition'}
        )
        fig.update_layout(
            title='Prevalence of Health Conditions Among Donors',
            xaxis_title='Percentage of Donors (%)',
            yaxis_title='Health Condition',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Eligibility Distribution by Gender")
        # Create a Plotly stacked bar chart
        gender_elig = pd.crosstab(data['Genre'], data['ELIGIBILITE AU DON.'], normalize='index') * 100
        
        fig = go.Figure()
        for col in gender_elig.columns:
            fig.add_trace(go.Bar(
                x=gender_elig.index,
                y=gender_elig[col],
                name=col,
                marker_color='green' if col == 'Eligible' else 'orange' if col == 'Temporairement Non-eligible' else 'red'
            ))
        
        fig.update_layout(
            barmode='stack',
            title='Eligibility Status by Gender',
            xaxis_title='Gender',
            yaxis_title='Percentage (%)',
            legend_title='Eligibility Status',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if hemoglobin_column:
            st.subheader("Hemoglobin Levels by Eligibility")
            # Create a Plotly box plot
            fig = px.box(
                data,
                x="ELIGIBILITE AU DON.",
                y=hemoglobin_column,
                color="ELIGIBILITE AU DON.",
                labels={hemoglobin_column: 'Hemoglobin Level (g/dL)', "ELIGIBILITE AU DON.": 'Eligibility Status'},
                category_orders={"ELIGIBILITE AU DON.": ["Eligible", "Temporairement Non-eligible", "Definitivement Non-eligible"]}
            )
            fig.update_layout(
                title='Hemoglobin Levels by Eligibility Status',
                xaxis_title='Eligibility Status',
                yaxis_title='Hemoglobin Level (g/dL)',
                height=500
            )
            # Add threshold lines for men and women
            fig.add_shape(type="line", x0=-0.5, x1=2.5, y0=13.0, y1=13.0,
                        line=dict(color="blue", width=2, dash="dash"), name="Min Male (13.0 g/dL)")
            fig.add_shape(type="line", x0=-0.5, x1=2.5, y0=12.5, y1=12.5,
                        line=dict(color="red", width=2, dash="dash"), name="Min Female (12.5 g/dL)")
            fig.add_annotation(x=2.5, y=13.0, text="Min Male (13.0 g/dL)", showarrow=False, xshift=80)
            fig.add_annotation(x=2.5, y=12.5, text="Min Female (12.5 g/dL)", showarrow=False, xshift=80)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a more detailed hemoglobin analysis by gender
            st.subheader("Hemoglobin Distribution by Gender")
            
            # Create a plotly histogram with KDE
            fig = go.Figure()
            
            # Add histograms for male and female
            male_hb = data[data["Genre"] == "Homme"][hemoglobin_column].dropna()
            female_hb = data[data["Genre"] == "Femme"][hemoglobin_column].dropna()
            
            fig.add_trace(go.Histogram(
                x=male_hb,
                opacity=0.7,
                name="Male",
                marker_color='blue',
                xbins=dict(size=0.5),
                histnorm='probability density'
            ))
            
            fig.add_trace(go.Histogram(
                x=female_hb,
                opacity=0.7,
                name="Female",
                marker_color='red',
                xbins=dict(size=0.5),
                histnorm='probability density'
            ))
            
            # Add KDE lines
            import numpy as np
            from scipy import stats
            
            def kde_curve(data, color):
                if len(data) > 1:
                    kde = stats.gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    y_values = kde(x_range)
                    return go.Scatter(x=x_range, y=y_values, mode='lines', line=dict(color=color), name=f'KDE')
                return None
            
            male_kde = kde_curve(male_hb, 'darkblue')
            if male_kde:
                fig.add_trace(male_kde)
                
            female_kde = kde_curve(female_hb, 'darkred')
            if female_kde:
                fig.add_trace(female_kde)
            
            # Add threshold vertical lines
            fig.add_shape(type="line", x0=13.0, x1=13.0, y0=0, y1=0.8,
                        line=dict(color="blue", width=2, dash="dash"))
            fig.add_shape(type="line", x0=12.5, x1=12.5, y0=0, y1=0.8,
                        line=dict(color="red", width=2, dash="dash"))
            
            fig.add_annotation(x=13.0, y=0.8, text="Min Male (13.0 g/dL)", 
                            showarrow=True, arrowhead=1, ax=40, ay=-40)
            fig.add_annotation(x=12.5, y=0.8, text="Min Female (12.5 g/dL)", 
                            showarrow=True, arrowhead=1, ax=-40, ay=-40)
            
            fig.update_layout(
                title='Hemoglobin Distribution by Gender',
                xaxis_title='Hemoglobin Level (g/dL)',
                yaxis_title='Density',
                legend_title='Gender',
                height=500,
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Dataset 'data_2019_cleaned.csv' not found.")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")

# Tab 3: Dataset Exploration
with tab3:
    st.header("Explore the Dataset")
    st.markdown("Insights into the dataset used for training the model.")

    try:
        df = pd.read_csv("data/data_2019_cleaned.csv")
        
        # Create a column name mapping to handle potential column name variations
        hemoglobin_column = None
        if "Taux d'hemoglobine" in df.columns:
            hemoglobin_column = "Taux d'hemoglobine"
        elif "Taux_dhemoglobine" in df.columns:
            hemoglobin_column = "Taux_dhemoglobine"
        
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        st.subheader("Basic Statistics")
        st.write(df.describe())

        st.subheader("Class Distribution")
        # Create a pie chart for class distribution
        class_counts = df["ELIGIBILITE AU DON."].value_counts()
        class_percentages = class_counts / class_counts.sum() * 100
        
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title='Distribution of Eligibility Status',
            color=class_counts.index,
            color_discrete_map={
                'Eligible': 'green',
                'Temporairement Non-eligible': 'orange',
                'Definitivement Non-eligible': 'red'
            },
            hole=0.3
        )
        
        fig.update_traces(textinfo='percent+label', textposition='inside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        if hemoglobin_column:
            st.subheader("Hemoglobin Distribution")
            # Create a histogram with density curve
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df[hemoglobin_column].dropna(),
                opacity=0.6,
                name="All Donors",
                marker_color='teal',
                xbins=dict(size=0.5),
                histnorm='probability density'
            ))
            
            # Add KDE curve
            import numpy as np
            from scipy import stats
            
            kde_data = df[hemoglobin_column].dropna()
            if len(kde_data) > 1:
                kde = stats.gaussian_kde(kde_data)
                x_range = np.linspace(kde_data.min(), kde_data.max(), 100)
                y_values = kde(x_range)
                fig.add_trace(go.Scatter(
                    x=x_range, 
                    y=y_values, 
                    mode='lines', 
                    line=dict(color='darkblue', width=2), 
                    name='Density'
                ))
            
            fig.update_layout(
                title='Hemoglobin Distribution',
                xaxis_title='Hemoglobin Level (g/dL)',
                yaxis_title='Density',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Age Distribution")
        # Create a histogram for age distribution by eligibility status
        fig = px.histogram(
            df[(df["Age"] >= 16) & (df["Age"] <= 80)],
            x="Age",
            color="ELIGIBILITE AU DON.",
            barmode="overlay",
            opacity=0.7,
            nbins=20,
            color_discrete_map={
                'Eligible': 'green',
                'Temporairement Non-eligible': 'orange',
                'Definitivement Non-eligible': 'red'
            }
        )
        
        # Add vertical lines for min and max age
        fig.add_shape(type="line", x0=18, x1=18, y0=0, y1=df["Age"].value_counts().max(),
                    line=dict(color="gray", width=2, dash="dash"))
        fig.add_shape(type="line", x0=65, x1=65, y0=0, y1=df["Age"].value_counts().max(),
                    line=dict(color="gray", width=2, dash="dash"))
        
        fig.add_annotation(x=18, y=df["Age"].value_counts().max(), text="Min Age (18)", 
                        showarrow=True, arrowhead=1, ax=-40, ay=-40)
        fig.add_annotation(x=65, y=df["Age"].value_counts().max(), text="Max Age (65)", 
                        showarrow=True, arrowhead=1, ax=40, ay=-40)
        
        fig.update_layout(
            title='Age Distribution by Eligibility Status',
            xaxis_title='Age',
            yaxis_title='Count',
            legend_title='Eligibility Status',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("BMI Distribution")
        # Calculate BMI
        df['BMI'] = df['Poids'] / ((df['Taille'] / 100) ** 2)
        valid_bmi = df[(df['BMI'] >= 15) & (df['BMI'] <= 40)]
        
        # Create a histogram for BMI distribution by eligibility status
        fig = px.histogram(
            valid_bmi,
            x="BMI",
            color="ELIGIBILITE AU DON.",
            barmode="overlay",
            opacity=0.7,
            nbins=25,
            color_discrete_map={
                'Eligible': 'green',
                'Temporairement Non-eligible': 'orange',
                'Definitivement Non-eligible': 'red'
            }
        )
        
        # Add vertical lines for BMI categories
        fig.add_shape(type="line", x0=18.5, x1=18.5, y0=0, y1=valid_bmi["BMI"].value_counts().max(),
                    line=dict(color="orange", width=2, dash="dash"))
        fig.add_shape(type="line", x0=25, x1=25, y0=0, y1=valid_bmi["BMI"].value_counts().max(),
                    line=dict(color="green", width=2, dash="dash"))
        fig.add_shape(type="line", x0=30, x1=30, y0=0, y1=valid_bmi["BMI"].value_counts().max(),
                    line=dict(color="red", width=2, dash="dash"))
        
        fig.add_annotation(x=18.5, y=valid_bmi["BMI"].value_counts().max(), text="Underweight (<18.5)", 
                        showarrow=True, arrowhead=1, ax=-60, ay=-40)
        fig.add_annotation(x=25, y=valid_bmi["BMI"].value_counts().max() * 0.8, text="Normal (18.5-25)", 
                        showarrow=True, arrowhead=1, ax=0, ay=-40)
        fig.add_annotation(x=30, y=valid_bmi["BMI"].value_counts().max() * 0.6, text="Obese (>30)", 
                        showarrow=True, arrowhead=1, ax=40, ay=-40)
        
        fig.update_layout(
            title='BMI Distribution by Eligibility Status',
            xaxis_title='BMI',
            yaxis_title='Count',
            legend_title='Eligibility Status',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Create an interactive correlation heatmap
        st.subheader("Feature Correlations")
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        # Drop columns with too many NaNs
        numeric_df = numeric_df.dropna(axis=1, thresh=len(numeric_df)*0.5)
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            labels=dict(color="Correlation")
        )
        fig.update_layout(
            title='Correlation Matrix of Numeric Features',
            height=700,
            width=700
        )
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("Dataset file 'data_2019_cleaned.csv' not found.")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Developed by Team CodeFlow | Powered by Streamlit, FastAPI, and scikit-learn")

# Add a debug section hidden behind an expander
with st.expander("Debugging Information", expanded=False):
    st.write("### API Connection Information")
    if st.button("Test API Connection"):
        api_status, api_message = check_api_status(api_url)
        if api_status:
            st.success(f"API Status: {api_message}")
            # Try a simple health check that includes model loaded status
            try:
                health_response = requests.get(f"{api_url}/health", timeout=5)
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    st.json(health_data)
                else:
                    st.error(f"API health check failed: {health_response.status_code}")
            except Exception as e:
                st.error(f"Error during health check: {str(e)}")
        else:
            st.error(f"API Status: {api_message}")
    
    st.write("### Session State")
    if st.checkbox("Show session state"):
        st.write(st.session_state)
    
    st.write("### Last API Response")
    if st.session_state.api_response:
        st.json(st.session_state.api_response)
    elif st.session_state.api_error:
        st.error(f"Last API Error: {st.session_state.api_error}")
    else:
        st.info("No API response recorded yet.")