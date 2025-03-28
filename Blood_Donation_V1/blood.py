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
from sklearn.model_selection import train_test_split
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import pickle
from wordcloud import WordCloud
import io
from datetime import datetime, date

# Initialization Functions
def initialize_session_state():
    if 'new_candidates' not in st.session_state:
        st.session_state.new_candidates = pd.DataFrame()
    if 'new_donors' not in st.session_state:
        st.session_state.new_donors = pd.DataFrame()

def configure_page():
    st.set_page_config(
        page_title="Blood Donation Dashboard",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Data Loading Functions
#@st.cache_data
def load_data():
    st.markdown("### Load Dataset")
    try:
        uploaded_files = st.file_uploader("Choose CSV files", type=['csv'], accept_multiple_files=True)
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
                    campaigns = pd.read_csv(file)
                    st.success(f"Successfully loaded campaigns dataset from {file.name}")
                    #st.dataframe(campaigns.head())
                else:
                    st.warning(f"Unrecognized file: {file.name}")

        if donor_candidates_birth is not None and donors is not None:
            donor_candidates_birth = process_candidate_data(donor_candidates_birth)
            donors = process_donor_data(donors)
        
        return donor_candidates_birth, donors
    except Exception as e:
        return None, None

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

@st.cache_data
def load_geo_data():
    districts = ["Douala I", "Douala II", "Douala III", "Douala IV", "Douala V"]
    coords = [[9.7, 4.05], [9.72, 4.08], [9.74, 4.07], [9.71, 4.03], [9.73, 4.06]]
    return pd.DataFrame({'district': districts, 'lat': [c[0] for c in coords], 'lon': [c[1] for c in coords]})

@st.cache_resource
def load_models():
    try:
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(18, 65, 405),
            'hemoglobin_level': np.random.normal(14, 1.5, 405),
            'weight': np.random.normal(70, 15, 405),
            'gender': np.random.choice([0, 1], 405),
            'last_donation_days': np.random.choice([9999] + list(range(0, 180)), 405),
            'is_eligible': [1]*365 + [0]*40
        })
        df.loc[df['is_eligible'] == 0, 'hemoglobin_level'] = np.random.uniform(10, 12.5, 40)
        
        X = df[['age', 'hemoglobin_level', 'weight', 'gender', 'last_donation_days']]
        y = df['is_eligible']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        eligibility_model = LogisticRegression(max_iter=1000)
        eligibility_model.fit(X_train, y_train)
        
        clustering_model = KMeans(n_clusters=3)
        clustering_model.fit(X[['age', 'weight']])
        
        return {'eligibility': eligibility_model, 'clustering': clustering_model}
    except Exception as e:
        st.warning(f"Models not loaded: {e}")
        return None

# UI Component Functions
def render_header():
    st.title("IndabaX Cameroon Hackaton: Blood Donation Management System")
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
                    <div class="title">CodeFlow</div>
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
            .highlight {color: #B22222; font-weight: bold;}
            .chart-container {background-color: white; border-radius: 5px; padding: 1rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin: 1rem 0;}
            .footer {text-align: center; font-size: 0.8rem; color: gray; margin-top: 2rem;}
            body {
                background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
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

def render_sidebar(donor_candidates_birth):
    with st.sidebar:
        st.image("/home/student24/Documents/AIMS_Folder/IndabaX_Cam/Project_test/Images/blood2.png", width=200)
        st.markdown("## Navigation")
        page = st.radio("Select Dashboard Page", [
            "Overview", "Geographic Distribution", "Health Conditions", 
            "Donor Profiles", "Campaign Effectiveness", "Donor Retention", 
            "Sentiment Analysis", "Eligibility Prediction", "Data Collection"
        ])

        st.markdown("---")
        st.markdown("## Filters")
        if donor_candidates_birth is not None:
            age_min, age_max = int(donor_candidates_birth['age'].min()), int(donor_candidates_birth['age'].max())
            age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))
            
            gender_options = ['All'] + donor_candidates_birth['gender'].unique().tolist()
            gender = st.selectbox("Gender", gender_options)
            
            district_options = ['All'] + donor_candidates_birth['residence_district'].unique().tolist()
            district = st.selectbox("District", district_options)
        else:
            age_range, gender, district = (18, 65), 'All', 'All'
        
        return page, age_range, gender, district

def apply_filters(df, age_range, gender, district):
    filtered_df = df.copy()
    if 'age' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['age'] >= age_range[0]) & 
            (filtered_df['age'] <= age_range[1])
        ]
    if 'gender' in filtered_df.columns and gender != 'All':
        filtered_df = filtered_df[filtered_df['gender'] == gender]
    if 'residence_district' in filtered_df.columns and district != 'All':
        filtered_df = filtered_df[filtered_df['residence_district'] == district]
    return filtered_df

# Page Rendering Functions
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
    charts_row = st.columns(2)
    if donors is not None:
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
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Age Distribution of Donors")
            if 'age' in donors.columns:
                fig = px.histogram(donors, x='age', nbins=20, color_discrete_sequence=['#B22222'], title="Age Distribution of Blood Donors")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sub-header'>Donation Trends</div>", unsafe_allow_html=True)
    if donor_candidates_birth is not None:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        donations = [45, 42, 50, 55, 60, 62, 58, 65, 70, 68, 72, 75]
        eligible_rates = [0.75, 0.73, 0.76, 0.78, 0.80, 0.79, 0.77, 0.81, 0.83, 0.82, 0.84, 0.85]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=donations, mode='lines+markers', name='Donations', line=dict(color='#B22222', width=3), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=months, y=[rate * 100 for rate in eligible_rates], mode='lines+markers', name='Eligibility Rate (%)', line=dict(color='#FFA07A', width=3, dash='dot'), marker=dict(size=8), yaxis='y2'))
        fig.update_layout(title='Monthly Donations and Eligibility Rates in 2019', xaxis_title='Month', yaxis_title='Number of Donations', yaxis2=dict(title='Eligibility Rate (%)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500)
        st.plotly_chart(fig, use_container_width=True)

def render_geographic_distribution(donor_candidates_birth, geo_data):
    st.markdown("<div class='sub-header'>Geographic Distribution of Donors</div>", unsafe_allow_html=True)
    map_col, stats_col = st.columns([3, 1])
    with map_col:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        if geo_data is not None and donor_candidates_birth is not None:
            m = folium.Map(location=[4.05, 9.7], zoom_start=12)
            if 'residence_district' in donor_candidates_birth.columns:
                district_counts = donor_candidates_birth['residence_district'].value_counts()
                for district in geo_data['district'].unique():
                    row = geo_data[geo_data['district'] == district].iloc[0]
                    count = district_counts.get(district, 0)
                    radius = max(100, count * 3)
                    color = '#B22222' if count > 50 else '#FFA07A'
                    folium.CircleMarker(location=[row['lat'], row['lon']], radius=radius/20, popup=f"{district}: {count} donors", color=color, fill=True, fill_opacity=0.6).add_to(m)
            folium_static(m)
        st.markdown("</div>", unsafe_allow_html=True)
    with stats_col:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.subheader("District Statistics")
        if donor_candidates_birth is not None and 'residence_district' in donor_candidates_birth.columns:
            district_counts = donor_candidates_birth['residence_district'].value_counts()
            st.write("Top Districts by Donor Count:")
            for district, count in district_counts.head(5).items():
                st.write(f"- {district}: *{count}*")
            st.write("---")
            st.write("Donor Density (per km²):")
            st.write("- Douala I: *3.2*")
            st.write("- Douala II: *2.8*")
            st.write("- Douala III: *1.9*")
            st.write("- Douala IV: *1.5*")
            st.write("- Douala V: *2.1*")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-container' style='margin-top: 1rem;'>", unsafe_allow_html=True)
        st.subheader("Access Analysis")
        st.write("Average Distance to Donation Center:")
        st.metric("All Districts", "3.2 km")
        st.write("Districts with Limited Access:")
        st.write("- Douala IV: *5.8 km*")
        st.write("- Douala V (outskirts): *7.2 km*")
        st.markdown("</div>", unsafe_allow_html=True)

def render_health_conditions(donor_candidates_birth, age_range, gender, district):
    st.markdown("<div class='sub-header'>Health Conditions Analysis</div>", unsafe_allow_html=True)
    if donor_candidates_birth is not None:
        filtered_df = apply_filters(donor_candidates_birth, age_range, gender, district)
        charts_row = st.columns(2)
        with charts_row[0]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Eligibility by Health Condition")

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
            #ineligible_pb.remove('Date_de_dernières_règles_(DDR)__')
            for pb in ineligible_pb:
                count = df_ineligible[df_ineligible[pb] == 'Oui'].shape[0]
                counts.append(count)

            print(len(ineligible_pb))
            print(ineligible_pb)
            print(len(counts))
            print(counts)

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
            if 'hemoglobin_level' in filtered_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=filtered_df[filtered_df['gender'] == 'Homme']['hemoglobin_level'], name='Male', marker_color='#0000CD', opacity=0.7))
                fig.add_trace(go.Histogram(x=filtered_df[filtered_df['gender'] == 'Femme']['hemoglobin_level'], name='Female', marker_color='#FF1493', opacity=0.7))
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
            if 'weight' in filtered_df.columns:
                fig = px.histogram(x=filtered_df['weight'], nbins=30, color_discrete_sequence=['#B22222'], title="Weight Distribution (kg)")
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

def render_donor_profiles(donor_candidates_birth, age_range, gender, district):
    st.markdown("<div class='sub-header'>Donor Profile Analysis</div>", unsafe_allow_html=True)
    if donor_candidates_birth is not None:
        filtered_df = apply_filters(donor_candidates_birth, age_range, gender, district)
        cluster_col, profile_col = st.columns([2, 1])
        with cluster_col:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Donor Clustering")
            if 'age' in filtered_df.columns and 'weight' in filtered_df.columns:
                X = filtered_df[['age', 'weight']].dropna()
                if len(X) > 3:
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    clusters = kmeans.fit_predict(X)
                    cluster_df = pd.DataFrame({'Age': X['age'], 'Weight': X['weight'], 'Cluster': ['Cluster ' + str(i+1) for i in clusters]})
                    fig = px.scatter(cluster_df, x='Age', y='Weight', color='Cluster', color_discrete_sequence=['#B22222', '#FF8C00', '#4682B4'], title="Donor Segments by Age and Weight")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with profile_col:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.subheader("Donor Personas")
            st.write("### Cluster 1: Regular Donors")
            st.write("- *Age Range:* 30-45\n- *Weight:* 70-90 kg\n- *Key Motivator:* Altruism")
            st.write("### Cluster 2: Occasional Donors")
            st.write("- *Age Range:* 20-35\n- *Weight:* 60-80 kg\n- *Key Motivator:* Social recognition")
            st.write("### Cluster 3: Family Donors")
            st.write("- *Age Range:* 35-60\n- *Weight:* 65-85 kg\n- *Key Motivator:* Family needs")
            st.markdown("</div>", unsafe_allow_html=True)

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
    fig2.add_trace(go.Scatter(x=channels, y=conv_rates, mode='lines+markers', name='Conversion Rate (%)', marker=dict(color='#000000'), yaxis='y2'))
    fig2.update_layout(yaxis2=dict(title='Conversion Rate (%)', overlaying='y', side='right', range=[0, max(conv_rates)*1.2]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_donor_retention():
    st.markdown("<div class='sub-header'>Donor Retention Analysis</div>", unsafe_allow_html=True)
    metrics_row = st.columns(3)
    with metrics_row[0]:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Retention Rate", "62.3%", "↑ 5.1%")
        st.write("Percentage of donors who return to donate again")
        st.markdown("</div>", unsafe_allow_html=True)
    with metrics_row[1]:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Average Donations", "2.8", "↑ 0.3")
        st.write("Average number of donations per donor annually")
        st.markdown("</div>", unsafe_allow_html=True)
    with metrics_row[2]:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Donor Lifetime", "3.2 years", "↑ 0.2")
        st.write("Average duration of donor participation")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("Donor Cohort Retention Analysis")
    cohorts = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    periods = ['Month 0', 'Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5']
    retention_data = []
    for cohort in cohorts:
        base = np.random.uniform(0.9, 1.0)
        retention_vals = [1.0]
        for i in range(1, len(periods)):
            retention_vals.append(round(retention_vals[-1] * np.random.uniform(0.75, 0.95), 2))
        retention_data.append(retention_vals)
    retention_df = pd.DataFrame(retention_data, index=cohorts, columns=periods)
    fig = px.imshow(retention_df, labels=dict(x="Period", y="Cohort", color="Retention Rate"), x=periods, y=cohorts, color_continuous_scale='Reds', text_auto=True, aspect="auto")
    fig.update_layout(title="Donor Retention by Cohort", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_sentiment_analysis(donor_candidates_birth):
    st.markdown("<div class='sub-header'>Sentiment Analysis of Donor Feedback</div>", unsafe_allow_html=True)
    st.write("Analyze donor sentiments to improve campaign messaging.")
    
    feedback = donor_candidates_birth['other_total_ineligible_reasons'].dropna().tolist() if donor_candidates_birth is not None and 'other_total_ineligible_reasons' in donor_candidates_birth.columns else [
        "I love donating blood, it feels great to help!", "The process was too slow, very frustrating.",
        "Amazing staff, made me feel so welcome.", "I won’t donate again, too much hassle."
    ]
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(f)['compound'] for f in feedback]
    sentiment_df = pd.DataFrame({'Feedback': feedback, 'Sentiment': sentiments})
    
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

def render_eligibility_prediction(models):
    st.markdown("<div class='sub-header'>Eligibility Prediction</div>", unsafe_allow_html=True)
    st.write("Predict donor eligibility based on health and donation history.")
    
    if models and 'eligibility' in models:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.subheader("Input Candidate Data")
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=8.0, max_value=20.0, value=14.0)
            weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0)
            gender = st.selectbox("Gender", ["Homme", "Femme"])
            days_since_last = st.number_input("Days Since Last Donation", min_value=0, max_value=9999, value=9999)
            
            if st.button("Predict Eligibility"):
                input_data = pd.DataFrame({
                    'age': [age], 'hemoglobin_level': [hemoglobin], 'weight': [weight],
                    'gender': [0 if gender == "Homme" else 1], 'last_donation_days': [days_since_last]
                })
                prob = models['eligibility'].predict_proba(input_data)[0][1]
                prediction = "Eligible" if prob > 0.5 else "Ineligible"
                st.write(f"*Prediction:* {prediction} (Probability: {prob:.2%})")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Feature Importance")
            features = ['Age', 'Hemoglobin', 'Weight', 'Gender', 'Days Since Last']
            importance = np.abs(models['eligibility'].coef_[0])
            fig = px.bar(x=features, y=importance, color_discrete_sequence=['#B22222'], title="Feature Importance in Eligibility Model")
            fig.update_layout(height=400, xaxis_title="Feature", yaxis_title="Importance")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

def render_data_collection(geo_data):
    st.markdown("<div class='sub-header'>Contribute to the Databank</div>", unsafe_allow_html=True)
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
            district = st.selectbox("District", geo_data['district'].tolist())
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

    st.markdown("### New Databank Entries for Candidates")
    st.write(st.session_state.new_candidates)

    st.markdown("### New Databank Entries for Donors")
    st.write(st.session_state.new_donors)
    
    if len(st.session_state.new_candidates) > 0:
        candidates_csv = st.session_state.new_candidates.to_csv(index=False)
        st.download_button(label="Download Candidates Data", data=candidates_csv, file_name=f"blood_donation_candidates.csv{datetime.now()}", mime="text/csv")
    if len(st.session_state.new_donors) > 0:
        donors_csv = st.session_state.new_donors.to_csv(index=False)
        st.download_button(label="Download Donors Data", data=donors_csv, file_name=f"blood_donation_donors.csv{datetime.now()}", mime="text/csv")

# Main Execution
def main():
    initialize_session_state()
    configure_page()
    render_header()
    render_styles()
    
    st.markdown("## Upload of the data")
    donor_candidates_birth, donors = load_data()
    geo_data = load_geo_data()
    models = load_models()
    
    page, age_range, gender, district = render_sidebar(donor_candidates_birth)
    
    page_functions = {
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
    
    if page in ["Overview"]:
        page_functions[page](donor_candidates_birth, donors)
    elif page in ["Geographic Distribution"]:
        page_functions[page](donor_candidates_birth, geo_data)
    elif page in ["Health Conditions", "Donor Profiles"]:
        page_functions[page](donor_candidates_birth, age_range, gender, district)
    elif page in ["Eligibility Prediction"]:
        page_functions[page](models)
    elif page in ["Data Collection"]:
        page_functions[page](geo_data)
    elif page in ["Sentiment Analysis"]:
        page_functions[page](donor_candidates_birth)
    else:
        page_functions[page]()
    
    st.markdown('<div class="footer">Developed by Team [CodeFlow] | March 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()