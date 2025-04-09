import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

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
# Custom CSS
st.markdown("""
    <style>
    .important-field { background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #1e90ff; }
    .important-label { color: #1e90ff; font-weight: bold; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
    .stButton>button:hover { background-color: #45a049; }
    </style>
""", unsafe_allow_html=True)


# Charger le jeu de données
df = pd.read_csv("data/data_2019_cleaned.csv")
df.drop(columns=["Taille"], inplace=True)
df_eligible = df.copy()

df_eligible = df_eligible[df_eligible["ELIGIBILITE AU DON."] == "Eligible"].copy()

# Dictionnaire de regroupement des professions
profession_groups = {
    "Chaudronnier": "Industrie & BTP", "Soudeur": "Industrie & BTP", "Mecanicien": "Industrie & BTP",
    "Macon": "Industrie & BTP", "Technicien en metallurgie": "Industrie & BTP", "Technicien genie civil": "Industrie & BTP",
    "Electrotechnicien": "Industrie & BTP", "Electricien en batiment": "Industrie & BTP", "Peintre": "Industrie & BTP",
    "Plombier": "Industrie & BTP", "Menuisier": "Industrie & BTP", "Carreleur": "Industrie & BTP",
    "Decorateur batiment": "Industrie & BTP", "Technicien etancheite": "Industrie & BTP",

    "Commercant": "Commerce & Entrepreneuriat", "Negociant bois": "Commerce & Entrepreneuriat",
    "Vendeur": "Commerce & Entrepreneuriat", "Entrepreneur": "Commerce & Entrepreneuriat",
    "Business man": "Commerce & Entrepreneuriat", "Trader": "Commerce & Entrepreneuriat",
    "Agent commercial": "Commerce & Entrepreneuriat", "Agent immobilier": "Commerce & Entrepreneuriat",
    "Restaurateur": "Commerce & Entrepreneuriat", "Magasinier": "Commerce & Entrepreneuriat",

    "Chauffeur": "Transport & Logistique", "Machiniste": "Transport & Logistique", "Docker": "Transport & Logistique",
    "Grutier": "Transport & Logistique", "Logisticien": "Transport & Logistique", "Transitaire": "Transport & Logistique",
    "Agent fret airport": "Transport & Logistique", "Gestionnaire de vols": "Transport & Logistique",
    "Conducteur": "Transport & Logistique",

    "Secretaire comptable": "Administration & Gestion", "Comptable": "Administration & Gestion",
    "Comptable financier": "Administration & Gestion", "Gestionnaire": "Administration & Gestion",
    "Assistant administratif": "Administration & Gestion", "Auditeur interne": "Administration & Gestion",
    "Administrateur": "Administration & Gestion", "Charge de clientele": "Administration & Gestion",
    "Charge de communication": "Administration & Gestion", "Intendant infirmier superieur": "Administration & Gestion",

    "Informaticien": "Informatique & Telecommunications", "Developpeur en informatique": "Informatique & Telecommunications",
    "Technicien reseaux telecoms": "Informatique & Telecommunications", "Analyste-programmeur": "Informatique & Telecommunications",
    "Informaticien de reseau": "Informatique & Telecommunications", "Infographe": "Informatique & Telecommunications",
    "Content manager": "Informatique & Telecommunications",

    "Enseignant": "Education & Recherche", "Professeur": "Education & Recherche",
    "Etudiant": "Education & Recherche", "Eleve": "Education & Recherche", "Stagiaire": "Education & Recherche",
    "Assistant juridique": "Education & Recherche",

    "Agent de securite": "Securite & Defense", "Chef de securite": "Securite & Defense",
    "Gendarme": "Securite & Defense", "Militaire": "Securite & Defense", "Brancardier": "Securite & Defense",

    "Medecin": "Sante & Social", "Personnel de sante": "Sante & Social",
    "Technicien de laboratoire": "Sante & Social", "Aide chirurgien": "Sante & Social",
    "Assistant infirmier": "Sante & Social", "Intendant infirmier superieur": "Sante & Social",

    "Beat maker": "Art & Culture", "Realisateur": "Art & Culture",
    "Chantre musicien": "Art & Culture", "Serigraphe": "Art & Culture", "Coiffeur": "Art & Culture",

    "Agent d'entretien": "Services & Autres", "Agent technique": "Services & Autres",
    "Technicien": "Services & Autres", "Electricien": "Services & Autres", "Hotelier": "Services & Autres",
    "Patissier": "Services & Autres", "Agent de maintenance industrielle": "Services & Autres",
    "Employe": "Services & Autres", "Operateur economique": "Services & Autres",

    "Sans emploi": "Sans emploi & Divers", "Pas precise": "Sans emploi & Divers"
}

# Appliquer le regroupement
df_eligible["Profession_Groupe"] = df_eligible["Profession"].map(profession_groups).fillna("Autres")

# Nettoyer les valeurs de la colonne "Taux d'hemoglobine"
df_eligible["Taux dhemoglobine"] = pd.to_numeric(df_eligible["Taux dhemoglobine"], errors='coerce')  # Convertir en float

# Sélection des variables utiles
features = ["Age", "Poids", "Niveau d'etude", "Genre", "Situation Matrimoniale (SM)", "Profession_Groupe","Taux dhemoglobine"]
df_eligible_clustering = df_eligible[features].copy()

# Encodage des variables catégorielles
label_encoders = {}
for col in ["Niveau d'etude", "Genre", "Situation Matrimoniale (SM)", "Profession_Groupe"]:
    le = LabelEncoder()
    df_eligible_clustering[col] = le.fit_transform(df_eligible_clustering[col])
    label_encoders[col] = le  # Stocker l'encodeur si besoin de décodage plus tard

# Normalisation des données pour le clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_eligible_clustering)

# Appliquer le clustering K-Means (choisir un nombre de clusters, ex: k=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_eligible["Cluster"] = kmeans.fit_predict(df_scaled)
# Affichage du nombre de personnes par cluster
print(df_eligible["Cluster"].value_counts())

# Trouver le cluster avec le plus grand nombre de personnes
cluster_counts = df_eligible["Cluster"].value_counts()
most_common_cluster = cluster_counts.idxmax()

print(f"Le cluster le plus représenté est : {most_common_cluster}")
print(cluster_counts)  # Afficher la répartition des clusters

# Filtrer uniquement les individus du cluster dominant
df_cluster_dominant = df_eligible[df_eligible["Cluster"] == most_common_cluster]

# Calculer les moyennes des variables numériques
mean_characteristics = df_cluster_dominant.mean(numeric_only=True)
print("\nCaractéristiques moyennes du cluster dominant :")
mean_characteristics = dict(mean_characteristics)
for key, value in mean_characteristics.items():
    mean_characteristics[key] = round(value)
print(mean_characteristics)
Mean_Keys = list(mean_characteristics.keys())
Mean_values = list(mean_characteristics.values())
categorical_features = ["Niveau d'etude", "Genre", "Situation Matrimoniale (SM)", "Profession_Groupe"]

# Trouver les valeurs dominantes pour chaque variable catégorielle
mode_characteristics = df_cluster_dominant[categorical_features].mode().iloc[0]

print("\nValeurs dominantes du cluster dominant :")
mode_characteristics = dict(mode_characteristics)
Mode_Keys = list(mode_characteristics.keys())
Mode_values = list(mode_characteristics.values())
print(mode_characteristics)

# Calculate ranges only
top_ages = df_cluster_dominant["Age"].value_counts().head(5)
age_values = top_ages.index.tolist()
age_range = f"{min(age_values)} - {max(age_values)} ans"

top_haemoglobin = df_cluster_dominant["Taux dhemoglobine"].round(1).value_counts().head(5)
hb_values = top_haemoglobin.index.tolist()
hb_range = f"{min(hb_values)} - {max(hb_values)} g/dL"

top_weights = df_cluster_dominant["Poids"].round(1).value_counts().head(5)
wt_values = top_weights.index.tolist()
wt_range = f"{min(wt_values)} - {max(wt_values)} kg"



if mean_characteristics is not None and mode_characteristics is not None:
    result_row = st.columns(1)
    with result_row[0]:
        # Display ranges on Streamlit
        st.markdown(f"""
        <div class="metric-container2">
            <h3>Minimal characteristics of an ideal donor</h3>
            <ul style="font-size: 1.3rem;">
                <li><strong>Âge</strong> : {age_range}</li>
                <li><strong>Taux d'hémoglobine</strong> : {hb_range}</li>
                <li><strong>Poids</strong> : {wt_range}</li>
                <li><strong>{Mode_Keys[0]}</strong> : {Mode_values[0]}</li>
                <li><strong>{Mode_Keys[2]}</strong> : {Mode_values[2]}</li>
                <li><strong>{Mode_Keys[3]}</strong> : {Mode_values[3]}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)



