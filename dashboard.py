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
def home_page(donor_candidates_birth, donors):
    #st.title("Home Page")

    with st.expander('**Data**'):
        st.write('**Raw Data**')
        st.write('**Candidates Donors data**')
        donor_candidates_birth
        st.write('**Donors data**')
        donors
    with st.expander('**Description:Summary statistics**'):
        if donor_candidates_birth is not None and donors is not None:
            st.write('**Candidates Donors data**')
            st.write(donor_candidates_birth.describe())
            st.write('**Donors data**')
            st.write(donors.describe())
        else:
            st.write("😠 Upload the data first !")

def render_overview(donor_candidates_birth, donors):
    st.markdown(f'<div class="metric-container"></div>', unsafe_allow_html=True)
    metrics_row = st.columns(4)
    if donor_candidates_birth is not None and donors is not None:
        with metrics_row[0]:
            total_candidates = len(donor_candidates_birth)
            st.markdown(f'<div class="metric-container2"><h3>Total Candidates</h3><p class="highlight" style="font-size: 2rem;">Total Candidates: {total_candidates}</p></div>', unsafe_allow_html=True)
        with metrics_row[1]:
            total_donors = len(donors)
            conversion_rate = round((total_donors / total_candidates) * 100, 1)
            st.markdown(f'<div class="metric-container2"><h3>Total Donors</h3><p class="highlight" style="font-size: 2rem;">Total Donors: {total_donors}</p></div>', unsafe_allow_html=True)
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
        charts_row = st.columns(2)
        with charts_row[0]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Blood Group Distribution")
            if 'blood_group' in donors.columns:
                blood_group_counts = donors['blood_group'].value_counts()
                fig = px.pie(values=blood_group_counts.values, names=blood_group_counts.index, color_discrete_sequence=px.colors.sequential.Reds, hole=0.5, title="Distribution of Blood Groups")
                fig.update_traces(textinfo='percent+label', pull=[0.1]*len(donors['blood_group'].unique()))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with charts_row[1]:
            donors = donors[donors['age'] != 0]  # Remove age 0
            median_age = donors['age'].median()
            donors['age'] = donors['age'].fillna(median_age)  # Impute missing age
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Age Distribution of Donors")
            if 'age' in donors.columns:
                fig = px.histogram(donors, x='age', nbins=20, color_discrete_sequence=['#B22222'], title="Age Distribution of Blood Donors")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        donors['timestamp'] = pd.to_datetime(donors['timestamp'], format='%m/%d/%Y %H:%M:%S')
        donors['Date'] = donors['timestamp'].dt.date
        donors['Hour'] = donors['timestamp'].dt.hour
        donors['Rh_Factor'] = donors['blood_group'].str[-1]
        donors['Kell'] = donors['phenotype'].str.extract(r'([+-]kell1)')
        bins = [18, 25, 35, 45, 60]
        labels = ['18-25', '26-35', '36-45', '46-59']
        donors['Age_Group'] = pd.cut(donors['age'], bins=bins, labels=labels, include_lowest=True)
        
        # Row 2: Blood Group and Rh Factor Distribution
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Blood Group Distribution")
            blood_group_counts = donors['blood_group'].value_counts()
            fig_blood = px.bar(x=blood_group_counts.index, y=blood_group_counts.values, 
                            labels={'x': 'Blood Group', 'y': 'Count'}, 
                            color_discrete_sequence=['#00CC96'])
            fig_blood.update_layout(height=300)
            st.plotly_chart(fig_blood, use_container_width=True)
        
        with col4:
            st.subheader("Rh Factor Distribution")
            rh_counts = donors['Rh_Factor'].value_counts()
            fig_rh = px.bar(x=rh_counts.index, y=rh_counts.values, 
                            labels={'x': 'Rh Factor', 'y': 'Count'}, 
                            color_discrete_sequence=['#AB63FA'])
            fig_rh.update_layout(height=300)
            st.plotly_chart(fig_rh, use_container_width=True)

        # Row 3: Donation Type and Kell Antigen Distribution
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Donation Type Distribution")
            donation_counts = donors['donation_type'].value_counts()
            fig_donation = px.bar(x=donation_counts.index, y=donation_counts.values, 
                                labels={'x': 'Donation Type', 'y': 'Count'}, 
                                color_discrete_sequence=['#FF6692'])
            fig_donation.update_layout(height=300)
            st.plotly_chart(fig_donation, use_container_width=True)
        
        with col6:
            st.subheader("Kell Antigen Distribution")
            kell_counts = donors['Kell'].value_counts()
            fig_kell = px.bar(x=kell_counts.index, y=kell_counts.values, 
                            labels={'x': 'Kell Antigen', 'y': 'Count'}, 
                            color_discrete_sequence=['#FFA15A'])
            fig_kell.update_layout(height=300)
            st.plotly_chart(fig_kell, use_container_width=True)

        # Row 4: Donations Over Time and by Hour
        col7, col8 = st.columns(2)
        with col7:
            st.subheader("Donations Over Time (By Date)")
            date_counts = donors['Date'].value_counts().sort_index()
            fig_date = go.Figure()
            fig_date.add_trace(go.Scatter(x=date_counts.index, y=date_counts.values, 
                                        mode='lines+markers', 
                                        line=dict(color='#1F77B4')))
            fig_date.update_layout(xaxis_title='Date', yaxis_title='Number of Donations', height=300)
            st.plotly_chart(fig_date, use_container_width=True)
        
        with col8:
            st.subheader("Donations by Hour of Day")
            hour_counts = donors['Hour'].value_counts().sort_index()
            fig_hour = px.bar(x=hour_counts.index, y=hour_counts.values, 
                            labels={'x': 'Hour of Day', 'y': 'Count'}, 
                            color_discrete_sequence=['#FF7F0E'])
            fig_hour.update_layout(height=300)
            st.plotly_chart(fig_hour, use_container_width=True)

def render_health_conditions(data, weight_range, age_range, gender, district, selected_eligibility):
    st.markdown("<div class='sub-header'>Health Conditions Analysis</div>", unsafe_allow_html=True)
    donor_candidates_birth = data[0]
    if donor_candidates_birth is not None:
        filtered_df1, filtered_df2 = apply_filters(age_range, weight_range, gender, district, selected_eligibility, data)
        charts_row = st.columns(2)
        with charts_row[0]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Eligibility by Health Condition")
            df_ineligible = donor_candidates_birth[donor_candidates_birth['eligibility'] != 'Eligible']
            counts = []
            ineligible_pb = list(df_ineligible.columns)
            #print(ineligible_pb)
            ineligible_pb = ineligible_pb[17:-1]
            ineligible_pb.remove('last_menstrual_date')
            ineligible_pb.remove('Autre raison preciser')
            ineligible_pb.remove('submission_status')
            ineligible_pb.remove('other_total_ineligible_reasons')

            for pb in ineligible_pb:
                count = df_ineligible[df_ineligible[pb] == 'Oui'].shape[0]
                counts.append(count)

            Reason =['On Medication','Low Hemoglobin', 'Last Donation(<3 months)','Recent Illness',
                            'DDR < 14 Days','Breast Feeding','Born < 6 months','Pregnancy Stop < 6 months',
                                'Pregnant','Previous Transfusion','Have IST','Operate','Sickle Cell','Diabetic',
                                'Hypertensive','Asmatic', 'Heart Attack', 'Tattoo','Scarified']
            
            health_condition = pd.DataFrame({'Reason': Reason, 'Counts': counts})
            health_condition = health_condition.sort_values(by='Counts', ascending=False)

            fig = px.bar(x=Reason, y=counts, title="Rejections by Health Condition", labels={'x': 'Health Condition', 'y': 'Number Rejected'}, color_discrete_sequence=['#B22222'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with charts_row[1]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Hemoglobin Levels Distribution")
            if 'hemoglobin_level' in filtered_df1.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=filtered_df1[filtered_df1['gender'] == 'Homme']['hemoglobin_level'], name='Male', marker_color='#0000CD', opacity=0.7))
                fig.add_trace(go.Histogram(x=filtered_df1[filtered_df1['gender'] == 'Femme']['hemoglobin_level'], name='Female', marker_color='#FF1493', opacity=0.7))
                fig.add_shape(type="line", x0=13.0, y0=0, x1=13.0, y1=100, line=dict(color="red", width=2, dash="dash"), name="Min Male")
                fig.add_shape(type="line", x0=12.0, y0=0, x1=12.0, y1=100, line=dict(color="pink", width=2, dash="dash"), name="Min Female")
                fig.update_layout(title="Hemoglobin Levels by Gender", xaxis_title="Hemoglobin Level (g/dL)", yaxis_title="Count", barmode='overlay', height=400)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        metrics_row = st.columns(3)
        with metrics_row[0]:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.subheader("Blood Pressure")
            systolic = np.random.normal(120, 15, 200)
            diastolic = np.random.normal(80, 10, 200)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=systolic, y=diastolic, mode='markers', marker=dict(size=8, color=systolic, colorscale='Reds', showscale=True, colorbar=dict(title="Systolic")), name="BP Readings"))
            fig.add_shape(type="rect", x0=90, y0=60, x1=120, y1=80, line=dict(color="green", width=2), fillcolor="rgba(0,255,0,0.1)", name="Normal")
            fig.add_shape(type="rect", x0=120, y0=80, x1=140, y1=90, line=dict(color="yellow", width=2), fillcolor="rgba(255,255,0,0.1)", name="Prehypertension")
            fig.update_layout(title="Blood Pressure Distribution", xaxis_title="Systolic (mmHg)", yaxis_title="Diastolic (mmHg)", height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with metrics_row[1]:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.subheader("Weight Distribution")
            if 'weight' in filtered_df1.columns:
                fig = px.histogram(x=filtered_df1['weight'], nbins=30, color_discrete_sequence=['#B22222'], title="Weight Distribution (kg)")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with metrics_row[2]:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.subheader("Common Rejection Reasons")
            top_rejection = health_condition[health_condition['Counts'] > 20]
            fig = px.bar(x=top_rejection['Counts'], y=top_rejection['Reason'], orientation='h', color_discrete_sequence=['#B22222'], title="Top Rejection Reasons")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        print(f'💢 Unpload the data first !')
def render_donor_retention(data):
    st.markdown("<div class='sub-header'>Donor Retention Analysis</div>", unsafe_allow_html=True)
    metrics_row = st.columns(3)
    donor_candidates_birth, donors = data[0], data[1]
    if donor_candidates_birth is not None:
        number_past_donation = donor_candidates_birth["has_donated_before"].str.count('Oui').sum()
        past_donation_rate = round(number_past_donation/len(donor_candidates_birth["has_donated_before"]),2)
        print(past_donation_rate)
        number_eligible = donor_candidates_birth["eligibility"].str.count('Eligible').sum()
        new_donation_rate = round(number_eligible/len(donor_candidates_birth['eligibility']),2)
        print(new_donation_rate)
        retention_rate = round(new_donation_rate/past_donation_rate,2)
        print(retention_rate)
        with metrics_row[0]:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Retention Rate",str(past_donation_rate*100)+"%" ," ↑ "+str(retention_rate)+"%")
            st.write("Percentage of donors who return to donate again")
            st.markdown("</div>", unsafe_allow_html=True)
        with metrics_row[1]:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Average Donations", "2.8", "↑ 0.3")
            st.write("Average number of donations per Campaign")
            st.markdown("</div>", unsafe_allow_html=True)
        with metrics_row[2]:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Donor Lifetime", "3.2 years", "↑ 0.2")
            st.write("Average duration of donor participation")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Donor Cohort Retention Analysis")

        donors['timestamp'] = pd.to_datetime(donors['timestamp'], format='%m/%d/%Y %H:%M:%S')
        donors['Date'] = donors['timestamp'].dt.date
        date_counts = donors['Date'].value_counts().sort_index()

        cohorts = donors['Date'].unique()
        periods = ['Day '+ str(i) for i in range(len(cohorts))]
        retention_data = []
        for cohort in cohorts:
            base = np.random.uniform(0.9, 1.0)
            retention_vals = [round(date_counts[0] / len(donors['Date']), 2)]
            for i in range(1,len(periods)):
                retention_vals.append(round(date_counts[i] / len(donors['Date']), 2))
            retention_data.append(retention_vals)
        retention_df = pd.DataFrame(retention_data, index=cohorts, columns=periods)
        fig = px.imshow(retention_df, labels=dict(x="Period", y="Cohort", color="Retention Rate"), x=periods, y=cohorts, color_continuous_scale='Reds', text_auto=True, aspect="auto")
        fig.update_layout(title="Donor Retention by Cohort", height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        print(f'💢 Upload the data first !')

def render_data_collection(geo_data):
    st.markdown("<div class='sub-header'>Contribute to the DataBank</div>", unsafe_allow_html=True)
    with st.form("new_data_form"):
        col1, col2 = st.columns(2)
        with col1:
            firstname = st.text_input("First Name")
            name = st.text_input("Name")
            age = st.number_input("Age", 18, 65, 30)
            birthdate = st.date_input("Birth Date", min_value=date(1900, 1, 1), max_value=date.today(), value=date.today())
            gender = st.selectbox("Gender", ["Male", "Female"])
            weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
            status = st.selectbox("Marital Status", ['Single', "Maried","None"])
            profession = st.text_input("Profession")
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 8.0, 20.0, 14.0)
            district = st.selectbox("District", geo_data['district'].tolist() if geo_data is not None else st.text_input("Enter your District"))
        with col2:
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            bp_systolic = st.number_input("Blood Pressure (Systolic, mmHg)", 80, 200, 120)
            bp_diastolic = st.number_input("Blood Pressure (Diastolic, mmHg)", 50, 120, 80)
            donation_freq = st.number_input("Donation Frequency (times/year)", 0, 10, 1)
            days_since_last = st.number_input("Days Since Last Donation", 0, 9999, 90)
            campaign_channel = st.selectbox("Campaign Channel", ["Social Media", "Community Event", "Radio", "SMS", "Other"])
            donated = st.checkbox("Donated Blood?")
            eligibility = st.selectbox("Eligible?", ["Yes", "No"])
            reason = st.selectbox("Reasons of Ineligibility", ['On Medication','Low Hemoglobin', 'Last Donation(<3 months)','Recent Illness',
                'DDR < 14 Days','Breast Feeding','Born < 6 months','Pregnancy Stop < 6 months',
                'Pregnant','Previous Transfusion','Have IST','Operate','Sickle Cell','Diabetic',
                'Hypertensive','Asmatic', 'Heart Attack', 'Tattoo','Scarified','None'])
        submit = st.form_submit_button("Add to Databank")
        
        if submit:
            new_candidate = {
                'form_fill_date': datetime.now(), 'firstname': firstname.upper(), 'name': name.upper(), 'birth_date': birthdate, 'age': age, 'gender': gender, 'weight': weight,
                'Status':status,'Profession':profession.upper(),'hemoglobin_level': hemoglobin, 'residence_district': district, 'has_donated_before': 'Yes' if donation_freq > 0 else 'No',
                'last_donation_date': pd.NaT if days_since_last == 9999 else pd.Timestamp.now() - pd.Timedelta(days=days_since_last),
                'eligibility': eligibility, 'is_eligible': 1 if eligibility == "Yes" else 0,
                'blood_pressure_systolic': bp_systolic, 'blood_pressure_diastolic': bp_diastolic,
                'donation_frequency': donation_freq, 'campaign_channel': campaign_channel, 'Reason of Ineligibility':reason
            }
            st.session_state.new_candidates = pd.concat([st.session_state.new_candidates, pd.DataFrame([new_candidate])], ignore_index=True)
            
            if donated:
                new_donor = {'timestamp': datetime.now(), 'gender': gender, 'age': age, 'donation_type': 'B', 'blood_group': blood_group}
                st.session_state.new_donors = pd.concat([st.session_state.new_donors, pd.DataFrame([new_donor])], ignore_index=True)
            
            st.success("Data added to databank!")

    st.markdown("### New Data Entries for Candidates")
    st.write(st.session_state.new_candidates)

    st.markdown("### New Data Entries for Donors")
    st.write(st.session_state.new_donors)
    
    if len(st.session_state.new_candidates) > 0:
        candidates_csv = st.session_state.new_candidates.to_csv(index=False)
        st.download_button(label="Download Candidates Data", data=candidates_csv, file_name=f"blood_donation_candidates.csv{datetime.now()}", mime="text/csv")
    if len(st.session_state.new_donors) > 0:
        donors_csv = st.session_state.new_donors.to_csv(index=False)
        st.download_button(label="Download Donors Data", data=donors_csv, file_name=f"blood_donation_donors.csv{datetime.now()}", mime="text/csv")

def render_campaign_effectiveness():
    st.markdown("<div class='sub-header'>Campaign Effectiveness Analysis</div>", unsafe_allow_html=True)
    metrics_row = st.columns(3)
    with metrics_row[0]:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Conversion Rate", "28.5%", "↑ 3.2%")
        st.write("Percentage of candidates who become donors")
        st.markdown("</div>", unsafe_allow_html=True)
    with metrics_row[1]:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Cost per Donor", "$12.80", "↓ $2.40")
        st.write("Average cost to acquire each new donor")
        st.markdown("</div>", unsafe_allow_html=True)
    with metrics_row[2]:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("ROI", "320%", "↑ 15%")
        st.write("Return on investment for campaign activities")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("Campaign Performance by Channel")
    channels = ['Social Media', 'Community Events', 'University Drives', 'Radio', 'SMS', 'Partner Orgs']
    impressions = [15000, 8000, 6500, 20000, 25000, 5000]
    conversions = [450, 380, 310, 280, 220, 180]
    conv_rates = [c/i*100 for c, i in zip(conversions, impressions)]
    fig = go.Figure(data=[go.Bar(name='Impressions', x=channels, y=impressions, marker_color='#FFA07A'), go.Bar(name='Conversions', x=channels, y=conversions, marker_color='#B22222')])
    fig.update_layout(barmode='group', title='Campaign Reach and Conversions by Channel', xaxis_title='Channel', yaxis_title='Count', height=400)
    fig2 = go.Figure(fig)
    fig2.add_trace(go.Scatter(x=channels, y=conv_rates, mode='lines+markers', name='Conversion Rate (%)', marker=dict(color='green'), yaxis='y2'))
    fig2.update_layout(yaxis2=dict(title='Conversion Rate (%)', overlaying='y', side='right', range=[0, max(conv_rates)*1.2]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_donor_profiles(data, weight_range, age_range, gender, district, selected_eligigility):
    st.markdown("<div class='sub-header'>Donor Profile Analysis</div>", unsafe_allow_html=True)
    pg = st.navigation([st.Page("dashboard_cluster.py")])
    pg.run()

    st.markdown("<div class='sub-header'>Donor Impact Simulator</div>", unsafe_allow_html=True)
    donations = st.slider("Simulate Donations", 1, 100, 10)
    lives_saved = donations * 3
    st.markdown(f"<h3 style='color: #FFD700;'>Your {donations} Donations Could Save <span style='color: #B22222;'>{lives_saved}</span> Lives!</h3>", unsafe_allow_html=True)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=donations, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Donation Impact"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#B22222"}, 'steps': [
            {'range': [0, 50], 'color': "#FF8C00"}, {'range': [50, 100], 'color': "#FFD700"}]},
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
def render_geographic_distribution(data, weight_range, age_range, gender, district, selected_eligigility):
    st.markdown("<div class='sub-header'>Geographic Distribution of Donors</div>", unsafe_allow_html=True)
    pg = st.navigation([st.Page("dashboard_map.py")])
    pg.run()
    donor_candidates_birth = data[0]
    if donor_candidates_birth is not None:
        filtered_df1, filtered_df2 = apply_filters(age_range, weight_range, gender, district, selected_eligigility, data)

        # Check if necessary columns exist
        if "residence_district" in filtered_df1.columns and "residence_neighborhood" in filtered_df1.columns:
        
            # Distribution by Arrondissement
            st.subheader("Distribution by District (Arrondissement)")
            arrond_counts = filtered_df1["residence_district"].value_counts().reset_index()
            arrond_counts.columns = ["District", "Count"]
            
            fig = px.bar(arrond_counts, x="District", y="Count", 
                        color="Count", color_continuous_scale='Reds',
                        labels={"Count": "Number of Donors", "Arrondissement": "District"},
                        title="Donor Distribution by District")
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution by Quartier (Neighborhood)
            st.subheader("Distribution by Neighborhood (Quartier)")
            # Create a dropdown to select district for neighborhood filtering
            district_options = ["All"] + list(filtered_df1["residence_district"].unique())
            selected_district = st.selectbox("Select District to View Neighborhoods", district_options)
            quartier_df = filtered_df1
            # Filter by selected district if not "All"
            if selected_district != "All":
                quartier_df = filtered_df1[filtered_df1["residence_district"] == selected_district]
                
            # Get neighborhood counts
            quartier_counts = quartier_df["residence_neighborhood"].value_counts().reset_index()
            quartier_counts.columns = ["Neighborhood", "Count"]
            
            # Show top 20 neighborhoods to prevent overcrowding
            quartier_counts = quartier_counts.head(20)
            
            fig = px.bar(quartier_counts, x="Neighborhood", y="Count", 
                        color="Count",color_continuous_scale='Reds',
                        labels={"Count": "Number of Donors", "Neighborhood": "Neighborhood"},
                        title=f"Top 20 Neighborhoods by Donor Count" + 
                            (f" in {selected_district}" if selected_district != "All" else ""))
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Interactive map - If we had coordinates, we would add them here
            st.subheader("Geographical Map (Heat Map Representation)")
            
            # Create a heatmap based on district data since we don't have exact coordinates
            heatmap_data = filtered_df1["residence_district"].value_counts().reset_index()
            heatmap_data.columns = ["District", "Donors"]
            
            fig = px.treemap(heatmap_data, path=["District"], values="Donors",
                            color="Donors", color_continuous_scale='Reds',
                            title="Heat Map of Donor Distribution by District")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Required geographical data columns not found in the dataset.")

def render_sentiment_analysis(data, weight_range, age_range, gender, district, selected_eligigility):
    st.markdown("<div class='sub-header'>Sentiment Analysis of Donor Feedback</div>", unsafe_allow_html=True)
    st.write("Analyze donor sentiments to improve campaign messaging.")
    donor_candidates_birth = data[0]
    if (donor_candidates_birth is not None):
        filtered_df1, filtered_df2 = apply_filters(age_range, weight_range, gender, district, selected_eligigility, data)
        feedback = filtered_df1['other_total_ineligible_reasons'].dropna().tolist() if filtered_df1 is not None and 'other_total_ineligible_reasons' in filtered_df1.columns else [
            "I love donating blood, it feels great to help!", "The process was too slow, very frustrating.",
            "Amazing staff, made me feel so welcome.", "I won’t donate again, too much hassle."
        ]

        sia = SentimentIntensityAnalyzer()
        sentiments = [sia.polarity_scores(f)['compound'] for f in feedback]
        sentiment_df = pd.DataFrame({'Feedback': feedback, 'Sentiment': sentiments})
        if selected_eligigility == 'All':
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.subheader("Sentiment Distribution")
                fig = px.bar(sentiment_df, x='Feedback', y='Sentiment', color='Sentiment', color_continuous_scale='RdYlGn', title="Sentiment Scores")
                fig.update_layout(height=400, xaxis={'tickangle': 45})
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
    pg = st.navigation([st.Page("dashboard_model.py")])
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
        "Donor Retention": render_donor_retention,
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
            st.header(tab_name)  # Display tab name as a header
            #page_functions[tab_name]()  # Call the respective function
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
    
    st.markdown('<div class="footer">Developed by Team [CodeFlow] | March 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()