import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import json
import random
import numpy as np
import pandas as pd


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

# Upload GeoJSON file
#uploaded_file = st.file_uploader("Choose a GeoJSON file", type="json")
# Specify the path to your JSON file
file_path = "data/geoBoundaries-CMR-ADM3.geojson"
# Open and read the JSON file
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# Predefined list of arrondissements
# Charger le jeu de données
df = pd.read_csv("data/candidates_2019_cleaned.csv")
quartiers = df["Quartier de Residence"].unique()
arrondissements = ["Douala III", 'Douala', 'Douala V', 'Douala I', 'Yaoundé', 'Douala II', 
                   'Douala IV', 'Bafoussam', 'Dschang', 'Buea', 'Non precise', 'Kribi', 
                   'Njombe', 'Tiko', 'Edéa', 'Manjo', 'West', 'Oyack', 'Deido', 'Douala VI', 
                   'Batie', 'Bomono ba mbegue', 'Meiganga', 'Sud ouest tombel', 
                   'Ngodi bakoko', 'LimbÃ©', 'Dcankongmondo', 'Boko']
# Function to generate a random color
def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'#{r:02x}{g:02x}{b:02x}'

# File uploader
file_path = "data/geoBoundaries-CMR-ADM3.geojson"
# Open and read the JSON file
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

if data is not None:
    # Read the GeoJSON file
    geojson_data = data
    
     # Create a color map for arrondissements
    color_map = {arr: random_color() for arr in arrondissements}
    
    # Filter features to only include predefined arrondissements
    filtered_features1 = [
        feature for feature in geojson_data['features'] 
        if feature['properties']['shapeName'] in arrondissements
    ]

    filtered_features2 = [
        feature for feature in geojson_data['features'] 
        if feature['properties']['shapeName'] in quartiers
    ]
    
    # Multiselect for places (only from filtered arrondissements)
    col1, col2 = st.columns(2)
    with col1:
        selected_places = st.multiselect(
            '**Select Arrondissements to focus on**', 
            arrondissements
        )
    with col2:
        selected_places = st.multiselect(
            '**Select Quaters to focus on**', 
            quartiers
        )
    
    # If no features found, show error
    if not filtered_features1 and not filtered_features2:
        st.error("No matching arrondissements found in the GeoJSON file.")
        st.error("No matching quaters found in the GeoJSON file.")
    else:
        # Calculate center coordinates
        all_coords = []
        for feature in filtered_features1 + filtered_features2:
            # Handle different geometry types (Polygon or MultiPolygon)
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
            elif feature['geometry']['type'] == 'MultiPolygon':
                coords = feature['geometry']['coordinates'][0][0]
            
            # Flatten coordinates and add to list
            all_coords.extend(coords)
        
        # Convert to numpy array for easier calculation
        coords_array = np.array(all_coords)
        
        # Calculate center
        center_lat = np.mean(coords_array[:, 1])
        center_lon = np.mean(coords_array[:, 0])
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
        
        # Create a marker cluster group
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add features to map
        for feature in filtered_features1 + filtered_features2:
            # Get the shapeName
            shape_name = feature['properties']['shapeName']
            
            # Check if this feature should be displayed
            if not selected_places or shape_name in selected_places:
                # Determine color
                fill_color = color_map.get(shape_name, '#888888')
                
                # Create style function with hover effect
                def style_function(feature):
                    return {
                        'fillColor': fill_color,
                        'color': 'black',
                        'weight': 2,
                        'fillOpacity': 0.5
                    }
                
                # Highlight style for hover
                def highlight_function(feature):
                    return {
                        'fillColor': fill_color,
                        'color': 'yellow',
                        'weight': 3,
                        'fillOpacity': 0.7
                    }
                
                # Add feature to map with hover and tooltip
                folium.GeoJson(
                    feature,
                    style_function=style_function,
                    highlight_function=highlight_function,
                    tooltip=folium.Tooltip(shape_name),
                ).add_to(m)
                
                # Add blood marker
                # Find centroid of the feature
                if feature['geometry']['type'] == 'Polygon':
                    coords = feature['geometry']['coordinates'][0]
                elif feature['geometry']['type'] == 'MultiPolygon':
                    coords = feature['geometry']['coordinates'][0][0]
                
                # Calculate centroid
                coords_array = np.array(coords)
                centroid_lat = np.mean(coords_array[:, 1])
                centroid_lon = np.mean(coords_array[:, 0])
                
                # Create blood drop marker
                folium.Marker(
                    location=[centroid_lat, centroid_lon],
                    popup=f"Blood Marker for {shape_name}",
                    icon=folium.Icon(color='red', icon='tint', prefix='fa')
                ).add_to(marker_cluster)
        
        # Display the map
        folium_static(m)