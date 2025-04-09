import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import pickle
from wordcloud import WordCloud
import io
from datetime import datetime, date
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve

# Initialization Functions
def initialize_session_state():
    if 'new_candidates' not in st.session_state:
        st.session_state.new_candidates = pd.DataFrame()
    if 'new_donors' not in st.session_state:
        st.session_state.new_donors = pd.DataFrame()

config = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": "custom_image",
        "height": 720,
        "width": 480,
        "scale": 6,
    }
}

# Configuration of the Web page
def configure_page():
    st.set_page_config(
        page_title="Blood Donation Dashboard",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# UI Component Functions
def render_header():
    col1,col2, spacer, col3 = st.columns([1,1,0.5,1])
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
    #with spacer:
    #    st.write("")
    with col3:
        st.image("Images/indabaX.jpeg", width= 200)

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

def process_candidate_data(df):
    candidates_columns = {
        "Date de remplissage de la fiche": "form_fill_date",
        "Date de naissance": "birth_date",
        "Age": "age",
        "Niveau d'etude": "education_level",
        "Genre": "gender",
        "Taille": "height",
        "Poids": "weight",
        "Situation Matrimoniale (SM)": "marital_status",
        "Profession": "profession",
        "Arrondissement de residence": "residence_district",
        "Quartier de Residence": "residence_neighborhood",
        "Nationalite": "nationality",
        "Religion": "religion",
        "A-t-il (elle) deja donne le sang": "has_donated_before",
        "Si oui preciser la date du dernier don.": "last_donation_date",
        "Taux d'hemoglobine": "hemoglobin_level",
        "ELIGIBILITE AU DON.": "eligibility",
        "Raison indisponibilite  [Est sous anti-biotherapie  ]": "ineligible_antibiotics",
        "Raison indisponibilite  [Taux d'hemoglobine bas ]": "ineligible_low_hemoglobin",
        "Raison indisponibilite  [date de dernier Don < 3 mois ]": "ineligible_recent_donation",
        "Raison indisponibilite  [IST recente (Exclu VIH, Hbs, Hcv)]": "ineligible_recent_sti",
        "Date de dernieres regles (DDR)": "last_menstrual_date",
        "Raison de l'indisponibilite de la femme [La DDR est mauvais si <14 jour avant le don]": "female_ineligible_menstrual",
        "Raison de l'indisponibilite de la femme [Allaitement ]": "female_ineligible_breastfeeding",
        "Raison de l'indisponibilite de la femme [A accoucher ces 6 derniers mois  ]": "female_ineligible_postpartum",
        "Raison de l'indisponibilite de la femme [Interruption de grossesse  ces 06 derniers mois]": "female_ineligible_miscarriage",
        "Raison de l'indisponibilite de la femme [est enceinte ]": "female_ineligible_pregnant",
        "Autre raisons,  preciser": "other_reasons",
        "Selectionner \"ok\" pour envoyer": "submission_status",
        "Raison de non-eligibilite totale  [Antecedent de transfusion]": "total_ineligible_transfusion_history",
        "Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]": "total_ineligible_hiv_hbs_hcv",
        "Raison de non-eligibilite totale  [Opere]": "total_ineligible_surgery",
        "Raison de non-eligibilite totale  [Drepanocytaire]": "total_ineligible_sickle_cell",
        "Raison de non-eligibilite totale  [Diabetique]": "total_ineligible_diabetes",
        "Raison de non-eligibilite totale  [Hypertendus]": "total_ineligible_hypertension",
        "Raison de non-eligibilite totale  [Asthmatiques]": "total_ineligible_asthma",
        "Raison de non-eligibilite totale  [Cardiaque]": "total_ineligible_heart_disease",
        "Raison de non-eligibilite totale  [Tatoue]": "total_ineligible_tattoo",
        "Raison de non-eligibilite totale  [Scarifie]": "total_ineligible_scarification",
        "Si autres raison preciser": "other_total_ineligible_reasons"
    }
    df.rename(columns=candidates_columns, inplace=True)
    
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    if 'eligibility' in df.columns:
        df['is_eligible'] = df['eligibility'].apply(
            lambda x: 1 if str(x).lower() in ['yes', 'oui', '1', 'true', 'eligible'] else 0
        )
    return df

def process_donor_data(df):
    donors_columns = {
        "Horodateur": "timestamp",
        "Sexe": "gender",
        "Age": "age",
        "Type de donation": "donation_type",
        "Groupe Sanguin ABO/Rhesus": "blood_group",
        "Phenotype": "phenotype"
    }
    df.rename(columns=donors_columns, inplace=True)
    return df

def render_sidebar():
    with st.sidebar:
        st.image("Images/codeflow.png", width=200)
            
        st.markdown("## **Navigation**")
        expander = st.expander("🗀 File Input")
        with expander:
            file_uploader_key = "file_uploader_{}".format(
                st.session_state.get("file_uploader_key", False)
            )

            uploaded_files = st.file_uploader(
                "Upload local files:",
                type=["csv","xls"],
                key=file_uploader_key,
                accept_multiple_files=True,
            )

            #uploaded_files = st.file_uploader("Choose CSV files", type=['csv','xls'], accept_multiple_files=True)
            donors, donor_candidates_birth, campaigns = None, None, None
            
            if uploaded_files:
                for file in uploaded_files:
                    if file.name.lower().startswith('donor'):
                        donors = pd.read_csv(file)
                        st.success(f"Successfully loaded donors dataset from {file.name}")
                        #st.dataframe(donors.head())
                    elif file.name.lower().startswith('candidates'):
                        donor_candidates_birth = pd.read_csv(file)
                        st.success(f"Successfully loaded candidates dataset from {file.name}")
                        #st.dataframe(donor_candidates_birth.head())
                    elif file.name.lower().startswith('campaign'):
                        campaigns = pd.read_excel(file)
                        st.success(f"Successfully loaded campaigns dataset from {file.name}")
                        #st.dataframe(campaigns.head())
                    else:
                        st.warning(f"Unrecognized file: {file.name}")

            if uploaded_files is not None:
                st.session_state["uploaded_files"] = uploaded_files
            if donor_candidates_birth is not None and donors is not None:
                donor_candidates_birth = process_candidate_data(donor_candidates_birth)
                donors = process_donor_data(donors)

        ## Definition of filters
        expander = st.expander("⚒ **Filters**")
        with expander:
            if donor_candidates_birth is not None:
                age_min, age_max = int(donor_candidates_birth['age'].min()), int(donor_candidates_birth['age'].max())
                age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

                weight_min, weight_max = int(donor_candidates_birth['weight'].min()), int(donor_candidates_birth['weight'].max())
                weight_range = st.slider("Weight Range", weight_min, weight_max, (weight_min, weight_max))
                
                gender_options = ['All'] + donor_candidates_birth['gender'].unique().tolist()
                gender = st.selectbox("Gender", gender_options)
                
                district_options = ['All'] + donor_candidates_birth['residence_district'].unique().tolist()
                district = st.selectbox("District", district_options)
                
                eligibility_options = ["All"] + list(donor_candidates_birth["eligibility"].unique())
                selected_eligibility = st.sidebar.selectbox("Eligibility Status", eligibility_options)
            else:
                age_range, weight_range, gender, district, selected_eligibility = 0, 0, 'All', 'All', 'All'
            
        data = [donor_candidates_birth, donors]
        return age_range, weight_range, gender, district, selected_eligibility, data

def apply_filters(age_range, weight_range, gender, district, selected_eligibility, df):
    if df[0] is not None and df[1] is not None:
        filtered_df1 = df[0].copy()
        filtered_df2 = df[1].copy()
        if 'age' in filtered_df1.columns:
            filtered_df1 = filtered_df1[
                (filtered_df1['age'] >= age_range[0]) & 
                (filtered_df1['age'] <= age_range[1])
            ]
        if 'weight' in filtered_df1.columns:
            filtered_df1 = filtered_df1[
                (filtered_df1['weight'] >= weight_range[0]) & 
                (filtered_df1['weight'] <= weight_range[1])
            ]
        if 'gender' in filtered_df1.columns and gender != 'All':
            filtered_df1 = filtered_df1[filtered_df1['gender'] == gender]
        if 'residence_district' in filtered_df1.columns and district != 'All':
            filtered_df1 = filtered_df1[filtered_df1['residence_district'] == district]
        if "eligibility" in filtered_df1.columns and selected_eligibility != "All":
            filtered_df1 = filtered_df1[filtered_df1["eligibility"] == selected_eligibility]
        print("Definition of Filters OK !")
        return filtered_df1, filtered_df2
    else:
        print("Data frame is empty !")
        return None, None

@st.cache_data
def load_geo_data(df):
    if df is not None:
        districts = df["residence_district"].unique()
        coords = [[9.7, 4.05], [9.72, 4.08], [9.74, 4.07], [9.71, 4.03], [9.73, 4.06]]
        #return pd.DataFrame({'district': districts, 'lat': [c[0] for c in coords], 'lon': [c[1] for c in coords]})
        return pd.DataFrame({'district': districts})
    else:
        return None

# Page Rendering Functions

## HOME FUNCTION
def home_page(donor_candidates_birth, donors):
    with st.expander('**Data**'):
        st.write('**Raw Data**')
        st.write('**Candidates Donors data**')
        st.dataframe(donor_candidates_birth)
        st.write('**Donors data**')
        st.dataframe(donors)

    with st.expander('**Summary Statistics (Quantitative Variables)**'):
        if donor_candidates_birth is not None and donors is not None:
            def exclude_binary(df):
                # Select numeric columns, excluding height
                numeric = df.select_dtypes(include='number').drop(columns=['height'], errors='ignore')

                # numeric = df.select_dtypes(include='number')
                return numeric.loc[:, numeric.nunique() > 2]  
            st.write('**Candidates Donors Data**')
            filtered_candidates = exclude_binary(donor_candidates_birth)
            st.dataframe(filtered_candidates.describe().T.style.format(precision=2))

        else:
            st.write("😠 Upload the data first!")

## OVERVIEW FUNCTION
def render_overview(donor_candidates_birth, donors):
    st.markdown(f'<div class="metric-container"></div>', unsafe_allow_html=True)
    metrics_row = st.columns(4)

    if donor_candidates_birth is not None and donors is not None:
        with metrics_row[0]:
            total_candidates = len(donor_candidates_birth)
            st.markdown(f'<div class="metric-container2"><h3>Total Candidates</h3><p class="highlight" style="font-size: 2rem;">{total_candidates}</p></div>', unsafe_allow_html=True)

        with metrics_row[1]:
            total_donors = len(donors)
            conversion_rate = round((total_donors / total_candidates) * 100, 1)
            st.markdown(f'<div class="metric-container2"><h3>Total Donors</h3><p class="highlight" style="font-size: 2rem;">{total_donors}</p></div>', unsafe_allow_html=True)

        with metrics_row[2]:
            if 'gender' in donors.columns:
                gender_counts = donors['gender'].value_counts()
                male_pct = round((gender_counts.get('M', 0) / total_donors) * 100, 1)
                female_pct = round((gender_counts.get('F', 0) / total_donors) * 100, 1)
                st.markdown(f'<div class="metric-container2"><h3>Gender Distribution</h3><p>Male: <span class="highlight">{male_pct}%</span></p><p> Female: <span class="highlight">{female_pct}%</span></p></div>', unsafe_allow_html=True)

        with metrics_row[3]:
            if 'donation_type' in donors.columns:
                donation_type_counts = donors['donation_type'].value_counts()
                voluntary_pct = round((donation_type_counts.get('B', 0) / total_donors) * 100, 1)
                family_pct = round((donation_type_counts.get('F', 0) / total_donors) * 100, 1)
                st.markdown(f'<div class="metric-container2"><h3>Donation Types</h3><p>Voluntary: <span class="highlight">{voluntary_pct}%</span></p><p>Family: <span class="highlight">{family_pct}%</span></p></div>', unsafe_allow_html=True)

    st.markdown("<div class='sub-header'>Key Insights</div>", unsafe_allow_html=True)

    if donors is not None:
        # Preprocessing
        donors = donors[donors['age'] != 0]
        donors['age'] = donors['age'].fillna(donors['age'].median())
        donors['timestamp'] = pd.to_datetime(donors['timestamp'], format='%m/%d/%Y %H:%M:%S')
        donors['Date'] = donors['timestamp'].dt.date
        donors['Hour'] = donors['timestamp'].dt.hour
        donors['Rh_Factor'] = donors['blood_group'].str[-1]
        donors['Kell'] = donors['phenotype'].str.extract(r'([+-]kell1)')
        bins = [18, 25, 35, 45, 60]
        labels = ['18-25', '26-35', '36-45', '46-59']
        donors['Age_Group'] = pd.cut(donors['age'], bins=bins, labels=labels, include_lowest=True)

        # ROW 1: Blood Group + Donation Type
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.subheader("Blood Group Distribution")
            blood_group_counts = donors['blood_group'].value_counts()
            fig_blood = px.bar(x=blood_group_counts.index, y=blood_group_counts.values,
                               labels={'x': 'Blood Group', 'y': 'Count'},
                               color_discrete_sequence=['#00CC96'])
            fig_blood.update_layout(height=300)
            st.plotly_chart(fig_blood, use_container_width=True)

        with row1_col2:
            st.subheader("Donation Type Distribution")
            donation_counts = donors['donation_type'].value_counts()
            fig_donation = px.bar(x=donation_counts.index, y=donation_counts.values,
                                  labels={'x': 'Donation Type', 'y': 'Count'},
                                  color_discrete_sequence=['#FF6692'])
            fig_donation.update_layout(height=300)
            st.plotly_chart(fig_donation, use_container_width=True)

        # ROW 2: Donation Over Time + Kell Antigen
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.subheader("Donations Over Time")
            date_counts = donors['Date'].value_counts().sort_index()
            fig_date = go.Figure()
            fig_date.add_trace(go.Scatter(x=date_counts.index, y=date_counts.values,
                                          mode='lines+markers',
                                          line=dict(color='#1F77B4')))
            fig_date.update_layout(xaxis_title='Date', yaxis_title='Number of Donations', height=300)
            st.plotly_chart(fig_date, use_container_width=True)

        with row2_col2:
            st.subheader("Kell Antigen Distribution")
            kell_counts = donors['Kell'].value_counts()
            fig_kell = px.bar(x=kell_counts.index, y=kell_counts.values,
                              labels={'x': 'Kell Antigen', 'y': 'Count'},
                              color_discrete_sequence=['#FFA15A'])
            fig_kell.update_layout(height=300)
            st.plotly_chart(fig_kell, use_container_width=True)

## HEATLTH CONDITION FUNCTION
def render_health_conditions(data, weight_range, age_range, gender, district, selected_eligibility):
    st.markdown("<div class='sub-header'>Health Conditions Analysis</div>", unsafe_allow_html=True)
    donor_candidates_birth = data[0]
    
    if donor_candidates_birth is not None:
        filtered_df1, filtered_df2 = apply_filters(age_range, weight_range, gender, district, selected_eligibility, data)
        
        # --- Compute rejection reasons
        df_ineligible = donor_candidates_birth[donor_candidates_birth['eligibility'] != 'Eligible']
        counts = []
        ineligible_pb = list(df_ineligible.columns)[17:-1]
        for col_to_remove in ['last_menstrual_date', 'Autre raison preciser', 'submission_status', 'other_total_ineligible_reasons']:
            if col_to_remove in ineligible_pb:
                ineligible_pb.remove(col_to_remove)

        Reason = ['On Medication','Low Hemoglobin', 'Last Donation(<3 months)','Recent Illness',
                    'DDR < 14 Days','Breast Feeding','Born < 6 months','Pregnancy Stop < 6 months',
                    'Pregnant','Previous Transfusion','Have IST','Operate','Sickle Cell','Diabetic',
                    'Hypertensive','Asmatic', 'Heart Attack', 'Tattoo','Scarified']

        for pb in ineligible_pb:
            count = df_ineligible[df_ineligible[pb] == 'Oui'].shape[0]
            counts.append(count)

        health_condition = pd.DataFrame({'Reason': Reason, 'Counts': counts})
        health_condition = health_condition.sort_values(by='Counts', ascending=False)

        # --- Row 1: Common Rejection Reasons + Hemoglobin Levels
        charts_row = st.columns(2)
        with charts_row[0]:
            st.subheader("Common Rejection Reasons")
            fig = px.bar(x=health_condition['Counts'], y=health_condition['Reason'],
                         orientation='h',
                         title="Top Rejection Reasons",
                         labels={'x': 'Number Rejected', 'y': 'Health Condition'},
                         color_discrete_sequence=['#B22222'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with charts_row[1]:
            st.subheader("Hemoglobin Levels Distribution")
            if 'hemoglobin_level' in filtered_df1.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=filtered_df1[filtered_df1['gender'] == 'Homme']['hemoglobin_level'],
                    name='Male', marker_color='#0000CD', opacity=0.7))
                fig.add_trace(go.Histogram(
                    x=filtered_df1[filtered_df1['gender'] == 'Femme']['hemoglobin_level'],
                    name='Female', marker_color='#FF1493', opacity=0.7))
                fig.add_shape(type="line", x0=13.0, y0=0, x1=13.0, y1=100, line=dict(color="red", width=2, dash="dash"))
                fig.add_shape(type="line", x0=12.0, y0=0, x1=12.0, y1=100, line=dict(color="pink", width=2, dash="dash"))
                fig.update_layout(title="Hemoglobin Levels by Gender", xaxis_title="Hemoglobin Level (g/dL)",
                                  yaxis_title="Count", barmode='overlay', height=400)
                st.plotly_chart(fig, use_container_width=True)

        # --- Row 2: Weight Distribution (enhanced binning)
        charts_row2 = st.columns(1)
        with charts_row2[0]:
            st.subheader("Weight Distribution")
            if 'weight' in filtered_df1.columns:
                fig = px.histogram(x=filtered_df1['weight'],
                                   nbins=50,
                                   color_discrete_sequence=['#B22222'],
                                   title="Weight Distribution (kg)")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("💢 Please upload the data first!")

## DATA COLLECTION FUNCTION
def render_data_collection(geo_data):
    st.markdown("<div class='sub-header'>Contribute to the DataBank</div>", unsafe_allow_html=True)
    #with st.form("new_data_form"):
    form_type = st.radio("Select Form Type", ["Donor Registration", "Blood Donation"])
    
    if form_type == "Donor Registration":
        st.subheader("New Donor Registration Form")
            
        with st.form("candidates_registration_form"):
        # Personal Information
            st.write("Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                firstname = st.text_input("First Name")
                name = st.text_input("Name")
                birth_date = st.date_input("Birth Date", min_value=date(1900, 1, 1), max_value=date.today(), value=date.today())
                gender = st.selectbox("Gender", ["Masculin", "Féminin"])
                height = st.number_input("Height (cm)", min_value=140, max_value=210)
                weight = st.number_input("Weight (kg)", min_value=40, max_value=150)
                
            with col2:
                education = st.selectbox("Education Level", ["Primaire", "Secondaire", "Supérieur"])
                marital_status = st.selectbox("Marital Status", ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf(ve)"])
                profession = st.text_input("Profession")
                
            # Location Information
            st.write("Location Information")
            col1, col2 = st.columns(2)
                
            with col1:
                district = st.selectbox("District", geo_data['district'].tolist() if geo_data is not None else st.text_input("Enter your District"))
                neighborhood = st.text_input("Neighborhood (Quartier)")
                
            with col2:
                nationality = st.text_input("Nationality")
                religion = st.selectbox("Religion", ["Chrétien", "Musulman", "Autre"])
                
            # Previous Donation History
            st.write("Donation History")
            previous_donation = st.radio("Previous Blood Donation", ["OUI", "NON"])
                
            if previous_donation == "OUI":
                last_donation_date = st.date_input("Last Donation Date", min_value=date(2020, 1, 1), max_value=date.today(), value=date.today())
                
            # Health Information
            st.write("Health Information")
                
            col1, col2, col3 = st.columns(3)
                
            with col1:
                transfusion_history = st.checkbox("History of Blood Transfusion")
                hiv_hbs_hcv = st.checkbox("Carrier of HIV, HBs, HCV")
                recent_surgery = st.checkbox("Recent Surgery")
                
            with col2:
                sickle_cell = st.checkbox("Sickle Cell Disease")
                diabetes = st.checkbox("Diabetes")
                hypertension = st.checkbox("Hypertension")
                
            with col3:
                asthma = st.checkbox("Asthma")
                heart_disease = st.checkbox("Heart Disease")
                recent_tattoo = st.checkbox("Recent Tattoo")
                scarification = st.checkbox("Scarification")
                
            # Female-specific Information
            if gender == "Féminin":
                st.write("Female-Specific Information")
                    
                col1, col2 = st.columns(2)
                    
                with col1:
                    last_period_date = st.date_input("Date of Last Menstrual", min_value=date(2020, 1, 1), max_value=date.today(), value=date.today())
                    pregnant = st.checkbox("Pregnant")
                    
                with col2:
                    breastfeeding = st.checkbox("Breastfeeding")
                    recent_childbirth = st.checkbox("Childbirth in the Last 6 Months")
                    miscarriage = st.checkbox("Miscarriage in the Last 6 Months")
                
            # Additional Information
            st.write("Additional Information")
            other_reasons = st.text_area("Other Medical Conditions")
            comment = st.text_area("Comments")
                
            # Submit Button
            submitted_candidates = st.form_submit_button("Register Candidate")
                
            if submitted_candidates:
                new_candidate = {
                    'form_fill_date': datetime.now(), 'firstname': firstname.upper(), 'name': name.upper(), 'birth_date': birth_date, 'gender': gender, 'weight': weight, 'height': height,
                    'education': education,'marital_status':marital_status,'Profession':profession.upper(),'nationality':nationality,'district': district,'quater': neighborhood,'religion':religion,
                    'previous_donation': previous_donation, 'Last_donation_date': last_donation_date if previous_donation == "Yes" else "NA",
                    'HIV_HBS_HCV': "Yes" if hiv_hbs_hcv else "No",'recent_surgery': "Yes"if recent_surgery else "No",'sickle_cell': "Yes" if sickle_cell else "No",
                    'diabetes': "Yes" if diabetes else "No", 'hypertension': "Yes" if hypertension else "No", 'asthma': "Yes" if asthma else "No", "heart_disease": "Yes" if heart_disease else "No",
                    'recent_tattoo': "Yes" if recent_tattoo else "No", 'scarification':"Yes" if scarification else "No", 'last_menstrual_date': last_period_date if gender == "Féminin" else "NA",
                    'pregnant': pregnant if gender == "Féminin" else "NA", 'breastfeeding': "Yes" if (gender == "Féminin" and breastfeeding) else "No", 'recent_childbirth': "Yes" if (gender == "Féminin" and recent_childbirth) else "NA",
                    'miscarriage': "Yes" if (gender == "Féminin" and miscarriage) else "NA", 'other_medical_reasons': other_reasons if len(other_reasons) != 0 else "None",
                    'comments': comment if len(comment) != 0 else "None"
                }
                st.session_state.new_candidates = pd.concat([st.session_state.new_candidates, pd.DataFrame([new_candidate])], ignore_index=True)
                st.success("Candidate registered successfully!")
                st.balloons()

        st.markdown("### New Data Entries for Candidates")
        st.write(st.session_state.new_candidates)

        if len(st.session_state.new_candidates) > 0:
            candidates_csv = st.session_state.new_candidates.to_csv(index=False)
            st.download_button(label="Download Candidates Data", data=candidates_csv, file_name=f"blood_donation_candidates.csv{datetime.now()}", mime="text/csv")
    
        
    else:  # Blood Donation Form
        st.subheader("Blood Donation Form")
            
        with st.form("donor_registration_form"):
            # Basic Information
            st.write("Donor Information")
                
            col1, col2 = st.columns(2)
                
            with col1:
                # In a real application, this would be a search for existing donors
                donor_id = st.text_input("Donor ID or Name")
                donation_date = st.date_input("Donation Date", min_value=date(2020, 1, 1), max_value=date.today(), value=date.today())
                birth_date_donor = st.date_input("Date of Birth")
                gender = st.selectbox("Gender", ["Masculin", "Féminin"])
            with col2:
                donation_type = st.selectbox("Donation Type", [
                    "B - Voluntary Donation",
                    "F - Family Donation"
                ])
                
            # Pre-donation Health Check
            st.write("Pre-donation Health Check")
                
            col1, col2 = st.columns(2)
                
            with col1:
                hemoglobin = st.number_input("Hemoglobin Level (g/dL)", min_value=10.0, max_value=20.0, step=0.1)
                
            with col2:
                under_antibiotics = st.checkbox("Currently Under Antibiotic Treatment")
                recent_sti = st.checkbox("Recent STI (excluding HIV, HBs, HCV)")
                
            # Blood Type Information
            st.write("Blood Type Information")
                
            col1, col2 = st.columns(2)
                
            with col1:
                blood_group = st.selectbox("Blood Group (ABO/Rh)", [
                    "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"
                ])
                
            with col2:
                phenotype = st.text_input("Phenotype (if known)")
                
            # Additional Notes
            notes = st.text_area("Additional Notes")
                
            # Submit Button
            submitted_donors = st.form_submit_button("Record Donation")
                
            if submitted_donors:
                new_donor = {'timestamp': datetime.now(),'donor_ID': donor_id, 'donation_date': donation_date, 'birth_date':birth_date_donor, 'gender': gender, 'donation_type': donation_type, 'hemoglobin': hemoglobin,'blood_group': blood_group,
                         'phenotype': phenotype if len(phenotype) > 1 else "Unknown", 'under_antibiotics': "Yes" if under_antibiotics else "No"
                         }
                st.session_state.new_donors = pd.concat([st.session_state.new_donors, pd.DataFrame([new_donor])], ignore_index=True)
            
                st.success("Blood donation recorded successfully!")
                st.balloons()   

        st.markdown("### New Data Entries for Donors")
        st.write(st.session_state.new_donors)
    
        if len(st.session_state.new_donors) > 0:
            donors_csv = st.session_state.new_donors.to_csv(index=False)
            st.download_button(label="Download Donors Data", data=donors_csv, file_name=f"blood_donation_donors.csv{datetime.now()}", mime="text/csv")

def render_campaign_effectiveness():
    st.markdown("<div class='sub-header'>Campaign Effectiveness Analysis</div>", unsafe_allow_html=True)
    
    metrics_row = st.columns(2)
    with metrics_row[0]:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Conversion Rate", "28.5%", "↑ 3.2%")
        st.write("Percentage of candidates who become donors")
        st.markdown("</div>", unsafe_allow_html=True)

    with metrics_row[1]:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Cost per Donor", "$12.80", "↓ $2.40")
        st.write("Average cost to recruit a donor (e.g. logistics, outreach)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

    channels = ['Social Media', 'Community Events', 'University Drives', 'Radio', 'SMS', 'Partner Orgs']
    impressions = [15000, 8000, 6500, 20000, 25000, 5000]
    conversions = [450, 380, 310, 280, 220, 180]
    conv_rates = [c/i*100 for c, i in zip(conversions, impressions)]

    fig = go.Figure(data=[
        go.Bar(name='Impressions', x=channels, y=impressions, marker_color='#FFA07A'),
        go.Bar(name='Conversions', x=channels, y=conversions, marker_color='#B22222')
    ])

    fig.update_layout(
        barmode='group',
        title='Campaign Reach and Conversions by Channel',
        xaxis_title='Channel',
        yaxis_title='Count',
        height=400
    )

    fig2 = go.Figure(fig)
    fig2.add_trace(go.Scatter(
        x=channels, y=conv_rates, mode='lines+markers',
        name='Conversion Rate (%)',
        marker=dict(color='green'),
        yaxis='y2'
    ))

    fig2.update_layout(
        yaxis2=dict(
            title='Conversion Rate (%)',
            overlaying='y',
            side='right',
            range=[0, max(conv_rates)*1.2]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


## Donor profile function
def render_donor_profiles(data, weight_range, age_range, gender, district, selected_eligigility):
    st.markdown("<div class='sub-header'>Donor Profile Analysis</div>", unsafe_allow_html=True)
    
    # Link to clustering page or donor profile breakdown
    pg = st.navigation([st.Page("dashboard_cluster.py")])
    pg.run()

## Geographic Distribution function
def render_geographic_distribution(data, weight_range, age_range, gender, district, selected_eligigility):
    st.markdown("<div class='sub-header'>Geographic Distribution of Donors</div>", unsafe_allow_html=True)
    pg = st.navigation([st.Page("dashboard_map.py")])
    pg.run()
    
    donor_candidates_birth = data[0]
    if donor_candidates_birth is not None:
        filtered_df1, filtered_df2 = apply_filters(age_range, weight_range, gender, district, selected_eligigility, data)

        # Check if necessary columns exist
        if "residence_district" in filtered_df1.columns and "residence_neighborhood" in filtered_df1.columns:

            #  Distribution by District (Arrondissement) - No color scale
            st.subheader("Distribution by District (Arrondissement)")
            arrond_counts = filtered_df1["residence_district"].value_counts().reset_index()
            arrond_counts.columns = ["District", "Count"]
            
            fig = px.bar(arrond_counts, x="District", y="Count",
                         labels={"Count": "Number of Donors", "District": "District"},
                         title="Donor Distribution by District")
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)

            #  Distribution by Quartier (Neighborhood)
            st.subheader("Distribution by Neighborhood (Quartier)")
            district_options = ["All"] + list(filtered_df1["residence_district"].unique())
            selected_district = st.selectbox("Select District to View Neighborhoods", district_options)

            quartier_df = filtered_df1
            if selected_district != "All":
                quartier_df = filtered_df1[filtered_df1["residence_district"] == selected_district]
                
            quartier_counts = quartier_df["residence_neighborhood"].value_counts().reset_index()
            quartier_counts.columns = ["Neighborhood", "Count"]
            quartier_counts = quartier_counts.head(20)  # limit to top 20

            fig = px.bar(quartier_counts, x="Neighborhood", y="Count",
                         labels={"Count": "Number of Donors", "Neighborhood": "Neighborhood"},
                         title=f"Top 20 Neighborhoods by Donor Count" + 
                               (f" in {selected_district}" if selected_district != "All" else ""))
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Required geographical data columns not found in the dataset.")

## Sentiment analysis
def render_sentiment_analysis(data, weight_range, age_range, gender, district, selected_eligigility):
    st.markdown("<div class='sub-header'>Sentiment Analysis of Donor Feedback</div>", unsafe_allow_html=True)
    st.write("Analyze donor sentiments to improve campaign messaging.")
    
    donor_candidates_birth = data[0]
    if donor_candidates_birth is not None:
        filtered_df1, filtered_df2 = apply_filters(age_range, weight_range, gender, district, selected_eligigility, data)
        
        feedback = filtered_df1['other_total_ineligible_reasons'].dropna().tolist() if filtered_df1 is not None and 'other_total_ineligible_reasons' in filtered_df1.columns else [
            "I love donating blood, it feels great to help!", "The process was too slow, very frustrating.",
            "Amazing staff, made me feel so welcome.", "I won’t donate again, too much hassle."
        ]

        sia = SentimentIntensityAnalyzer()
        sentiments = [sia.polarity_scores(f)['compound'] for f in feedback]
        sentiment_df = pd.DataFrame({'Feedback': feedback, 'Sentiment': sentiments})

        # Classify sentiments into Positive, Neutral, Negative
        sentiment_categories = []
        for score in sentiments:
            if score > 0.1:
                sentiment_categories.append('Positive')
            elif score < -0.1:
                sentiment_categories.append('Negative')
            else:
                sentiment_categories.append('Neutral')

        sentiment_df['Sentiment Category'] = sentiment_categories
        
        # Group by sentiment category and count occurrences
        sentiment_counts = sentiment_df['Sentiment Category'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            # Plotting the sentiment counts
            fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', color_discrete_sequence=['#3E9E5B', '#B22222', '#FF8C00'], 
                            title="Donor Feedback Sentiment Distribution")
            fig.update_traces(width=0.2)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.subheader("Word Cloud")
                text = " ".join(feedback)
                wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(text)
                plt.figure(figsize=(8, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("No feedback data available for other categories of eligible donors.")

def render_eligibility_prediction():
    pg = st.navigation([st.Page("test_dashboard.py")])
    pg.run()

def main():
    initialize_session_state()
    configure_page()
    render_header()
    render_styles()
    L = render_sidebar()
    age_range = L[0]
    weight_range = L[1]
    gender = L[2]
    district = L[3]
    selected_eligigility = L[4]
    data = L[5]

    if L is not None:
        donor_candidates_birth, donors = apply_filters(age_range, weight_range, gender, district, selected_eligigility, data)
        geo_data = load_geo_data(donor_candidates_birth)
    else:
        donor_candidates_birth, donors = None, None
 
    page_functions = {
        "Home Page": home_page,
        "Overview": render_overview,
        "Geographic Distribution": render_geographic_distribution,
        "Health Conditions": render_health_conditions,
        "Donor Profiles": render_donor_profiles,
        "Campaign Effectiveness": render_campaign_effectiveness,
        "Sentiment Analysis": render_sentiment_analysis,
        "Eligibility Prediction": render_eligibility_prediction,
        "Data Collection": render_data_collection
    }

    # Define the tab names
    tabs = list(page_functions.keys())  # Get the keys (tab names)

    # Create tabs
    tab_objects = st.tabs(tabs)

    # Loop through each tab and call the corresponding function
    for tab, tab_name in zip(tab_objects, tabs):
        with tab:
            if tab_name == "Home Page":
                page_functions[tab_name](donor_candidates_birth, donors)
            elif tab_name == "Overview":
                page_functions[tab_name](donor_candidates_birth, donors)
            elif tab_name in ["Geographic Distribution"]:
                page_functions[tab_name](data, weight_range, age_range, gender, district, selected_eligigility)
            elif tab_name == "Health Conditions":
                page_functions[tab_name](data, weight_range, age_range, gender, district, selected_eligigility)
            elif tab_name == "Donor Profiles":
                page_functions[tab_name](data, weight_range, age_range, gender, district, selected_eligigility)
            elif tab_name == "Eligibility Prediction":
                page_functions[tab_name]()
            elif tab_name == "Data Collection":
                page_functions[tab_name](geo_data)
            elif tab_name == "Sentiment Analysis":
                page_functions[tab_name](data, weight_range, age_range, gender, district, selected_eligigility)
            elif tab_name == "Donor Retention":
                page_functions[tab_name](data)
            elif tab_name in ["Campaign Effectiveness"]:
                page_functions[tab_name]()
            else:
                page_functions[tab_name]()
    
    st.markdown('<div class="footer">Developed by Team [CodeFlow] | 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()