import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify, session
import os
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib
import re
import uuid
import plotly.graph_objects as go
from datetime import timedelta
import math
from datetime import datetime
from meteostat import Point, Hourly
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

import time




# Configuration de l'application Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ma_cle_secrete'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['DATA_FOLDER'] = 'data'

# Créer les dossiers s'ils n'existent pas
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['DATA_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        


# ==============================================================================
# 🛠️ FONCTIONS UTILITAIRES GLOBALES 🛠️
# ==============================================================================
def is_numeric_or_datetime(series):
    """Vérifie si la colonne est de type numérique ou date/heure"""
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)
        

# ==============================================================================
# 🎨 DÉFINITION DES CARTES DE COULEURS ET DES CATÉGORIES
# ==============================================================================
COLOR_MAPS = {
    'temperature': {
        # Couleurs: Bleu (Froid), Vert (Confort), Rouge (Chaud)
        'colors': {'Froide (< 17 °C)': '#0070c0', 'Modérée (17 - 25 °C)': '#00b050', 'Chaude (> 25 °C)': '#ff0000', 'Inconnu': '#808080'},
        'categories': ['Froide (< 17 °C)', 'Modérée (17 - 25 °C)', 'Chaude (> 25 °C)', 'Inconnu']
    },
    'humidity': {
        # Couleurs: Jaune (Sec), Vert (Confort), Bleu (Humide)
        'colors': {'Faible (< 40 %)': '#ffc000', 'Modérée (40 - 60 %)': '#00b050', 'Élevée (> 60 %)': '#0070c0', 'Inconnu': '#808080'},
        'categories': ['Faible (< 40 %)', 'Modérée (40 - 60 %)', 'Élevée (> 60 %)', 'Inconnu']
    },
    'cov': {
        # Couleurs: Vert (Bon), Jaune (Modéré), Rouge (Mauvais)
        'colors': {'Bonne (≤ 50 ppb)': '#00b050', 'Modérée (50 - 100 ppb)': '#ffc000', 'Mauvaise (> 100 ppb)': '#ff0000', 'Inconnu': '#808080'},
        'categories': ['Bonne (≤ 50 ppb)', 'Modérée (50 - 100 ppb)', 'Mauvaise (> 100 ppb)', 'Inconnu']
    },
    'iqa': {
        # Couleurs: AQI standard (Vert, Jaune, Rouge, Violet)
        'colors': {'Bonne (≤ 50)': '#00b050', 'Modérée (50 - 100)': '#ffc000', 'Mauvaise (100 - 200)': '#ff0000', 'Très Mauvaise (> 200)': '#7030a0', 'Inconnu': '#808080'},
        'categories': ['Bonne (≤ 50)', 'Modérée (50 - 100)', 'Mauvaise (100 - 200)', 'Très Mauvaise (> 200)', 'Inconnu']
    },
    'pm': {
        # Couleurs: PM standard (Vert, Jaune, Rouge)
        'colors': {'Faible (< 10 µg/m³)': '#00b050', 'Modérée (10 - 45 µg/m³)': '#ffc000', 'Mauvaise (> 45 µg/m³)': '#ff0000', 'Inconnu': '#808080'},
        'categories': ['Faible (< 10 µg/m³)', 'Modérée (10 - 45 µg/m³)', 'Mauvaise (> 45 µg/m³)', 'Inconnu']
    },
    'co2': {
        # Couleurs: Vert (Confort), Rouge (Non-Confort)
        'colors': {'Confort (< 800 ppm)': '#00b050', 'Non-Confort (≥ 800 ppm)': '#ff0000', 'Inconnu': '#808080'},
        'categories': ['Confort (< 800 ppm)', 'Non-Confort (≥ 800 ppm)', 'Inconnu']
    },
    'nox': {
        # Couleurs: Vert (Bon), Rouge (Mauvais)
        'colors': {'Bonne (< 1 ppm)': '#00b050', 'Mauvaise (≥ 1 ppm)': '#ff0000', 'Inconnu': '#808080'},
        'categories': ['Bonne (< 1 ppm)', 'Mauvaise (≥ 1 ppm)', 'Inconnu']
    },
    'light': {
        # Couleurs: Jaune/Orange (Jour), Gris (Nuit)
        'colors': {'Journée (> 0 Lux)': '#ffc000', 'Nuit (0 Lux)': '#434343', 'Inconnu': '#808080'},
        'categories': ['Journée (> 0 Lux)', 'Nuit (0 Lux)', 'Inconnu']
    }
}
# ==============================================================================
# 🎨 DETERMINE LA VILLE
# ==============================================================================
def determine_city_and_month(filename):
    """
    Extrait la ville et le mois d'un nom de fichier au format 'ville-mois.csv'.
    """
    match = re.search(r'([a-zA-Z]+)-(\w+)\.csv$', filename)
    if match:
        city = match.group(1)
        month = match.group(2)
        return city.capitalize(), month.capitalize()
    return "Inconnue", "Inconnu"

# ==============================================================================
# 🎨 CONVERTIT LE TEMPS EN DATE ET HEURE
# ==============================================================================
def convert_to_datetime(df):
    """
    Tente de convertir la colonne de temps en spécifiant le format exact
    pour éviter les ambiguïtés entre les formats de date américains et européens.
    """
    time_col_name = None
    
    # Cherche une colonne qui s'appelle 'temps' ou qui contient 'temps'
    potential_cols = [col for col in df.columns if 'temps' in col.lower()]
    if not potential_cols:
        return df, None

    # Priorise la colonne nommée exactement 'temps'
    time_col_name = 'temps' if 'temps' in potential_cols else potential_cols[0]

    try:
        # On force le format jour/mois/année heure:minute:seconde
        # C'est la correction la plus importante.
        df[time_col_name] = pd.to_datetime(
            df[time_col_name], 
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )
    except Exception as e:
        app.logger.error(f"Échec de la conversion de la colonne '{time_col_name}' avec le format spécifié: {e}")
        # Si le format échoue, on peut tenter une détection automatique comme plan B
        try:
            df[time_col_name] = pd.to_datetime(
                df[time_col_name],
                errors='coerce',
                dayfirst=True
            )
        except Exception as e2:
            app.logger.error(f"La détection automatique a aussi échoué: {e2}")
            return df, None
        
    return df, time_col_name
# ==============================================================================
# 🎨 Charge le DataFrame depuis le fichier temporaire
# ==============================================================================
def get_df_from_session():
    
    if 'data_file' not in session:
        return None
    data_file = session['data_file']
    file_path = os.path.join(app.config['DATA_FOLDER'], data_file)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return None
# ==============================================================================
# 🎨 Sauvegarde le DataFrame dans le fichier temporaire de la session
# ==============================================================================
def update_df_in_session(df):
    
    data_filename = session.get('data_file')
    if data_filename:
        file_path = os.path.join(app.config['DATA_FOLDER'], data_filename)
        df.to_parquet(file_path)
        return True
    return False

# ==============================================================================
# 🎨 Crée une nouvelle colonne catégorielle pour la coloration des graphiques
# ==============================================================================
def create_color_categories(df, col_name, category_type):
    
    new_col_name = f'{col_name}_category'
    df[new_col_name] = 'Inconnu'
    
   
    map_data = COLOR_MAPS.get(category_type, {})
    color_map = map_data.get('colors', {})
    ordered_categories = map_data.get('categories', [])

   
    if category_type == 'temperature':
        df.loc[df[col_name] <= 17, new_col_name] = 'Froide (< 17 °C)'
        df.loc[(df[col_name] > 17) & (df[col_name] <= 25), new_col_name] = 'Modérée (17 - 25 °C)'
        df.loc[df[col_name] > 25, new_col_name] = 'Chaude (> 25 °C)'

    elif category_type == 'humidity':
        df.loc[df[col_name] < 40, new_col_name] = 'Faible (< 40 %)'
        df.loc[(df[col_name] >= 40) & (df[col_name] <= 60), new_col_name] = 'Modérée (40 - 60 %)'
        df.loc[df[col_name] > 60, new_col_name] = 'Élevée (> 60 %)'
    
    elif category_type == 'cov':
        df.loc[df[col_name] <= 50, new_col_name] = 'Bonne (≤ 50 ppb)'
        df.loc[(df[col_name] > 50) & (df[col_name] <= 100), new_col_name] = 'Modérée (50 - 100 ppb)'
        df.loc[df[col_name] > 100, new_col_name] = 'Mauvaise (> 100 ppb)'
    
    elif category_type == 'iqa':
        df.loc[df[col_name] <= 50, new_col_name] = 'Bonne (≤ 50)'
        df.loc[(df[col_name] > 50) & (df[col_name] <= 100), new_col_name] = 'Modérée (50 - 100)'
        df.loc[(df[col_name] > 100) & (df[col_name] <= 200), new_col_name] = 'Mauvaise (100 - 200)'
        df.loc[df[col_name] > 200, new_col_name] = 'Très Mauvaise (> 200)'
    
    elif category_type == 'pm':
        df.loc[df[col_name] < 10, new_col_name] = 'Faible (< 10 µg/m³)'
        df.loc[(df[col_name] >= 10) & (df[col_name] <= 45), new_col_name] = 'Modérée (10 - 45 µg/m³)'
        df.loc[df[col_name] > 45, new_col_name] = 'Mauvaise (> 45 µg/m³)'

    elif category_type == 'co2':
        df.loc[df[col_name] < 800, new_col_name] = 'Confort (< 800 ppm)'
        df.loc[df[col_name] >= 800, new_col_name] = 'Non-Confort (≥ 800 ppm)'

    elif category_type == 'nox':
        df.loc[df[col_name] < 1, new_col_name] = 'Bonne (< 1 ppm)'
        df.loc[df[col_name] >= 1, new_col_name] = 'Mauvaise (≥ 1 ppm)'
    
    elif category_type == 'light':
        df.loc[df[col_name] > 0, new_col_name] = 'Journée (> 0 Lux)'
        df.loc[df[col_name] == 0, new_col_name] = 'Nuit (0 Lux)'

    current_categories = [c for c in ordered_categories if c in df[new_col_name].unique()]
    
    df[new_col_name] = pd.Categorical(df[new_col_name], categories=current_categories, ordered=True)

    return df, new_col_name, color_map, ordered_categories

# ==============================================================================
# 🎨 CALCULE LE POINT DE ROSE
# ==============================================================================
def calculate_dew_point(temperature, humidity):
    """
    Calcule le point de rosée en utilisant l'approximation Magnus-Tetens.
    (Basé sur le calcul BME280 fourni par l'utilisateur, adapté en Python).
    """
    # (1) Saturation Vapor Pressure = ESGG(T)
    ratio = 373.15 / (273.15 + temperature)
    rhs = -7.90298 * (ratio - 1)
    rhs += 5.02808 * math.log10(ratio)
    rhs += -1.3816e-7 * (math.pow(10, (11.344 * (1 - 1 / ratio))) - 1)
    rhs += 8.1328e-3 * (math.pow(10, (-3.49149 * (ratio - 1))) - 1)
    rhs += math.log10(1013.246)
    
    vp = math.pow(10, rhs - 3) * (humidity / 100.0)
    
   
    A = 17.27
    B = 237.7
    
    # Calcul de alpha
    alpha = ((A * temperature) / (B + temperature)) + math.log(humidity / 100.0)
    
    # Calcul du point de rosée
    dew_point = (B * alpha) / (A - alpha)
    
    return dew_point

# ==============================================================================
# 🎨 PAGE INDEX
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')

# ==============================================================================
# 🛠️ ROUTE API PROXY (Pour l'autocomplétion sécurisée)
# ==============================================================================
@app.route('/api/experiments')
#@app.route('/api/experiments', methods=['GET'])
def get_experiments_proxy():
    """
    Récupère la liste fraîche depuis Eurosmart en contournant tous les caches possibles.
    """
    try:
        # 1. Timestamp pour tromper le cache du serveur distant
        timestamp = int(time.time())
        url = f"https://polluguard.eurosmart.fr/get_experiments?nocache={timestamp}"
        
        token = "IUFNIFN-9z84fSION@soi-efgzerg"
        headers = {
            "Authorization": f"Bearer {token}",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
        
        # 2. Appel externe
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # 3. On crée une réponse Flask avec des entêtes anti-cache stricts pour le navigateur
            flask_response = jsonify(data)
            flask_response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            flask_response.headers['Pragma'] = 'no-cache'
            flask_response.headers['Expires'] = '0'
            return flask_response
        else:
            app.logger.error(f"Erreur API Distante: {response.status_code}")
            return jsonify([]), response.status_code
            
    except Exception as e:
        app.logger.error(f"Erreur Route API: {e}")
        return jsonify({'error': str(e)}), 500

# ==============================================================================
# 🌍 FONCTION : REVERSE GEOCODING (Lat/Lon -> Ville)
# ==============================================================================
def get_city_from_coordinates(lat, lon):
    """
    Utilise OpenStreetMap (Nominatim) pour trouver la ville à partir des coordonnées.
    """
    try:
        # Il est important de donner un user_agent unique
        geolocator = Nominatim(user_agent="eurosmart_analytics_app")
        location = geolocator.reverse(f"{lat}, {lon}", language='fr', exactly_one=True)
        
        if location:
            address = location.raw.get('address', {})
            # On cherche la ville, ou le village, ou la commune
            city = address.get('city') or address.get('town') or address.get('village') or address.get('municipality')
            if city:
                return city
    except Exception as e:
        print(f"Erreur Geocoding: {e}")
    
    return None 


# ==============================================================================
# 📥 ROUTE D'UPLOAD
# ==============================================================================
@app.route('/upload-data')
def upload_data():
    return render_template('upload_data.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # 1. Récupération des entrées
    codes_input = request.form.get('code_exp')
    manual_city_fallback = request.form.get('city_exp', '').strip()
    
    if not codes_input:
        return jsonify({'error': 'Veuillez entrer au moins un code expérience.'}), 400
        
    delimiter = ';'    
    decimal = ','
    
    file_info_list = []
    all_data = pd.DataFrame()
    errors_log = []
    
    column_mapping = {
        '_HR': 'Humidité', '_Temp': 'Température', '_LUM': 'Lumière', '_VOC': 'COV',
        '_PM1': 'PM1', '_PM2': 'PM2_5', '_PM4': 'PM4', '_PM10': 'PM10',
        '_IQA': 'IQA', '_CO2': 'CO2', '_NOX': 'NOX', '_PA': 'Pression', 'Villes' : 'city' 
    }

    # --- ÉTAPE 1 : RÉCUPÉRER LA LISTE API ---
    api_experiments = []
    try:
        url = "https://polluguard.eurosmart.fr/get_experiments"
        token = "IUFNIFN-9z84fSION@soi-efgzerg"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers, timeout=5) # Timeout augmenté à 5s
        if response.status_code == 200:
            api_experiments = response.json()
    except Exception as e:
        app.logger.warning(f"API Eurosmart warning: {e}")

    # --- ÉTAPE 2 : TRAITEMENT DES CODES ---
    normalized_input = codes_input.replace(';', ',').replace(' ', ',')
    code_list = [c.strip() for c in normalized_input.split(',') if c.strip()]

    for i, code in enumerate(code_list):
        
        if i > 0:
            time.sleep(1.5) 

        try:
            # A. Téléchargement
            csv_url = f'https://version.eurosmart.fr/exp/polluguard_exp_{code}.csv'
            
            try:
                # low_memory=False aide pour les gros fichiers
                df_temp = pd.read_csv(csv_url, sep=delimiter, decimal=decimal, 
                                    storage_options={'User-Agent': 'Mozilla/5.0'},
                                    low_memory=False)
            except Exception:
                errors_log.append(f"{code}: Fichier introuvable ou erreur réseau.")
                continue
            
            if df_temp.empty:
                 errors_log.append(f"{code}: Fichier vide.")
                 continue

            # B. Détection Ville
            city_name = "Inconnu"
            
            # 1. Via API + Géolocalisation
            exp_info = next((item for item in api_experiments if item.get("CodeExp") == code), None)
            
            if exp_info and 'Latitude' in exp_info and 'Longitude' in exp_info:
                # Appel sécurisé à votre fonction get_city_from_coordinates
                try:
                    detected_city = get_city_from_coordinates(exp_info['Latitude'], exp_info['Longitude'])
                    if detected_city:
                        city_name = detected_city
                except Exception as e:
                    app.logger.error(f"Erreur Geocoding pour {code}: {e}")
            
            # 2. Via le nom du code
            if city_name == "Inconnu" and '_' in code:
                parts = code.split('_')
                # Vérifie que la première partie ressemble à une ville (lettres, >2 chars)
                if len(parts[0]) > 2 and not parts[0].isdigit():
                    city_name = parts[0].capitalize()
            
            # 3. Via saisie manuelle (seulement si on traite un seul code ou si c'est le seul moyen)
            if city_name == "Inconnu" and manual_city_fallback:
                city_name = manual_city_fallback
            
            # 4. Fallback final
            if city_name == "Inconnu":
                city_name = f"Exp_{code}"

            # C. Nettoyage
            df_temp.columns = df_temp.columns.astype(str)
            df_temp.columns = df_temp.columns.str.strip().str.replace(' ', '')

            rename_dict = {}
            for original, new in column_mapping.items():
                for col in df_temp.columns:
                    if original.lower() in col.lower() and col not in rename_dict.values():
                        rename_dict[col] = new
                        break
            if rename_dict: df_temp.rename(columns=rename_dict, inplace=True)
            
            # Suppression colonnes inutiles
            cols_to_drop = []
            time_col_found = False
            for col in df_temp.columns:
                if 'batterie' in col.lower(): cols_to_drop.append(col)
                elif 'temps' in col.lower():
                    if not time_col_found:
                        time_col_found = True
                        df_temp.rename(columns={col: 'temps'}, inplace=True)
                    else: cols_to_drop.append(col)
            if cols_to_drop: df_temp = df_temp.drop(columns=cols_to_drop, errors='ignore')
                
            # Date
            df_temp, time_col = convert_to_datetime(df_temp)
            if time_col: session['time_col'] = time_col
            
            # Ajout Méta
            df_temp['city'] = city_name
           
            
            # Remplissage NaN
            for col in df_temp.columns:
                if pd.api.types.is_numeric_dtype(df_temp[col]):
                    df_temp[col].fillna(df_temp[col].median(), inplace=True)

            file_info_list.append({'filename': code, 'city': city_name})
            
            # Concaténation
            all_data = pd.concat([all_data, df_temp], ignore_index=True)

        except Exception as e:
            app.logger.error(f"CRASH sur le code {code}: {e}")
            errors_log.append(f"{code}: Erreur interne ({str(e)}).")
            
    # --- 4. RETOUR ---
    if all_data.empty:  
        msg = "Échec du chargement."
        if errors_log: msg += " Détails : " + " | ".join(errors_log)
        return jsonify({'error': msg}), 400
            
    # Sauvegarde Parquet (Inchangé)
    data_filename = f'{uuid.uuid4()}.parquet'
    file_path = os.path.join(app.config['DATA_FOLDER'], data_filename)
    all_data.to_parquet(file_path)
    session['data_file'] = data_filename
    
    # --- MODIFICATION ICI : APERÇU INTELLIGENT ---
    # Au lieu de prendre juste les 5 premières lignes totales (head()),
    # on prend les 5 premières lignes de CHAQUE ville présente dans le fichier.
    
    try:
        # On groupe par 'city', on prend les 5 premiers de chaque, et on reset l'index pour l'affichage
        preview_df = all_data.groupby('city', sort=False).head(5).reset_index(drop=True)
        
        # Génération du HTML sans l'index (plus propre)
        preview_html = preview_df.to_html(classes='data-table table-striped table-bordered', index=False)
    except Exception as e:
        # Fallback au cas où (si la colonne city n'existe pas pour une raison obscure)
        app.logger.error(f"Erreur lors de l'aperçu groupé: {e}")
        preview_html = all_data.head(10).to_html(classes='data-table table-striped table-bordered')

    info = {        
        'rows': len(all_data),
        'columns': list(all_data.columns),
        'cols_info': all_data.dtypes.astype(str).to_dict(),
        'uploaded_files': file_info_list
    }
    
    msg = 'Données chargées avec succès.'
    if errors_log: msg += " (Attention: " + ", ".join(errors_log) + ")"
    
    return jsonify({'message': msg, 'data_info': info, 'preview': preview_html})       
# ==============================================================================
# 🎨 PAGE NETOYAGE
# ==============================================================================            
@app.route('/data-preparation')
def data_preparation():
    return render_template('data_preparation.html')

@app.route('/get_data_info')
def get_data_info():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400
    
    cols_info = df.dtypes.astype(str).to_dict()
    info = {
        'rows': len(df),
        'columns': list(df.columns),
        'cols_info': cols_info
    }
    preview_html = df.head().to_html(classes='data-table table-striped table-bordered')

    return jsonify({
        'data_info': info,
        'data_preview_html': preview_html
    })

@app.route('/get_unique_cities')
def get_unique_cities():
    df = get_df_from_session()
    if df is None:  
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400
        
    if 'city' not in df.columns:
        return jsonify({'cities': []}), 200
        
    unique_cities = df['city'].unique().tolist()
    return jsonify({'cities': unique_cities})

@app.route('/get_descriptive_stats')
def get_descriptive_stats():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400
        
    # Sélectionner uniquement les colonnes numériques pour les statistiques
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        return jsonify({'error': 'Le jeu de données ne contient pas de colonnes numériques.'}), 400

    # Calculer les statistiques descriptives
    stats_df = numeric_df.describe().transpose()
    
    # Renvoyer le tableau HTML des statistiques
    stats_html = stats_df.to_html(classes='table table-striped table-bordered', float_format='%.2f')
    
    return jsonify({'stats_html': stats_html})



@app.route('/calculate_dew_point')
def calculate_dew_point_route():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400

    # Vérifier si les colonnes nécessaires existent
    required_cols = ['Température', 'Humidité']
    if not all(col in df.columns for col in required_cols):
        return jsonify({'error': f"Les colonnes '{required_cols[0]}' et '{required_cols[1]}' sont requises pour ce calcul."}), 400

    # Appliquer le calcul du point de rosée
    try:
        df['point_de_rosee'] = df.apply(lambda row: calculate_dew_point(row['Température'], row['Humidité']), axis=1)
        update_df_in_session(df)
        
        # Retourner l'aperçu du DataFrame mis à jour
        preview_html = df.head().to_html(classes='data-table table-striped table-bordered')
        return jsonify({
            'message': 'Le point de rosée a été calculé avec succès et ajouté à vos données.',
            'data_preview_html': preview_html
        })
    except Exception as e:
        return jsonify({'error': f"Une erreur s'est produite lors du calcul : {str(e)}"}), 500

 
@app.route('/calculate_temp_dew_point_diff')
# Route pour calculer la différence entre la température et le point de rosée
@app.route('/calculate_temp_dew_point_diff', methods=['POST'])
def calculate_temp_dew_point_diff_route():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400

    # Vérifiez si les colonnes nécessaires existent
    if 'Température' not in df.columns or 'point_de_rosee' not in df.columns:
        return jsonify({'error': 'Les colonnes "température" ou "point_de_rosee" sont introuvables.'}), 400

    try:
        # Calcul de la différence et ajout de la nouvelle colonne
        df['diff_temp_rosee'] = df['Température'] - df['point_de_rosee']
        update_df_in_session(df)
        return jsonify({'message': 'Colonne "diff_temp_rosee" calculée et ajoutée avec succès.'})
    except Exception as e:
        return jsonify({'error': f"Erreur lors du calcul de la différence : {str(e)}"}), 500

# Route pour calculer les moyennes journalières et afficher le tableau
@app.route('/get_daily_dew_point_averages')
def get_daily_dew_point_averages():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400

    time_col = session.get('time_col')
    if not time_col or time_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        return jsonify({'error': 'La colonne de temps est nécessaire pour calculer les moyennes journalières.'}), 400

    # Colonnes à inclure dans le tableau
    cols_to_avg = ['Température', 'point_de_rosee', 'diff_temp_rosee']
    existing_cols = [col for col in cols_to_avg if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if not existing_cols:
        return jsonify({'error': 'Aucune colonne numérique pertinente (température, point_de_rosee, diff_temp_rosee) trouvée pour calculer les moyennes.'}), 400

    # Resampling des données par jour pour obtenir les moyennes
    df_resampled = df.set_index(time_col).resample('D')[existing_cols].mean().round(2).reset_index()

    # Formatter la colonne de temps pour l'affichage
    df_resampled[time_col] = df_resampled[time_col].dt.strftime('%Y-%m-%d')
    
    # Renommer la colonne de temps pour l'affichage
    df_resampled.rename(columns={time_col: 'Date'}, inplace=True)

    
    daily_avg_html = df_resampled.to_html(classes='data-table')    
    return jsonify({'daily_avg_html': daily_avg_html})



# ==============================================================================
# 🎨 PAGE VISUALISATION
# ==============================================================================
@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400
    
    plot_type = request.json.get('plotType')
    x_col = request.json.get('xCol')
    y_col = request.json.get('yCol')
    color_col = request.json.get('colorCol')
    color_type = request.json.get('colorType')
    city = request.json.get('city')

    if color_col == "":
        color_col = None
        color_type = None

    temp_df = df.copy()

    # ==============================================================================
    # 1. NETTOYAGE TEMPOREL (Uniquement pour 'line' et 'bar')
    # ==============================================================================
    is_time_series = False 

    # On applique le traitement SI c'est une LIGNE ou une BARRE
    if plot_type in ['line', 'bar']:
        if x_col in temp_df.columns:
            # 1. Conversion forcée en date
            temp_df[x_col] = pd.to_datetime(temp_df[x_col], errors='coerce')
            temp_df = temp_df.dropna(subset=[x_col])
            
            # 2. Tri chronologique (Indispensable pour l'axe temporel)
            temp_df = temp_df.sort_values(by=x_col)
            
            # 3. Suppression des doublons (Le fix du "200%")
            if 'city' in temp_df.columns:
                temp_df = temp_df.drop_duplicates(subset=[x_col, 'city'], keep='last')
            else:
                temp_df = temp_df.drop_duplicates(subset=[x_col], keep='last')
            
            is_time_series = True

    # ==============================================================================
    # 2. FILTRAGE PAR VILLE
    # ==============================================================================
    if city:
        if 'city' not in temp_df.columns:
            return jsonify({'error': "La colonne 'city' est introuvable."}), 400
        
        temp_df = temp_df[temp_df['city'] == city]
        
        if temp_df.empty:
            return jsonify({'error': f"Aucune donnée trouvée pour la ville sélectionnée ({city})."}), 400

    # ==============================================================================
    # 3. GESTION DES COULEURS
    # ==============================================================================
    color_map = None
    category_orders = {}

    if color_col and color_type:
        try:
            temp_df, new_color_col, color_map, ordered_categories = create_color_categories(temp_df, color_col, color_type)
            color_col = new_color_col
            category_orders[color_col] = ordered_categories
        except Exception as e:
            app.logger.error(f"Erreur couleur: {str(e)}")

    # Vérifications colonnes
    if x_col not in temp_df.columns:
        return jsonify({'error': f"La colonne '{x_col}' est introuvable."}), 400
    if y_col and y_col not in temp_df.columns:
        return jsonify({'error': f"La colonne '{y_col}' est introuvable."}), 400

    # ==============================================================================
    # 4. GÉNÉRATION DES GRAPHIQUES
    # ==============================================================================
    try:
        common_args = {
            'color_discrete_map': color_map,
            'category_orders': category_orders,
        }
        
        if plot_type == 'scatter':
            fig = px.scatter(temp_df, x=x_col, y=y_col, color=color_col, **common_args)
        
        elif plot_type == 'bar':
            # Maintenant temp_df est trié et nettoyé, donc pas de doublons empilés
            fig = px.bar(temp_df, x=x_col, y=y_col, color=color_col, barmode='group', **common_args)
            
        elif plot_type == 'line':
            if city:
                 fig = px.line(temp_df, x=x_col, y=y_col, **common_args)
            else:
                 fig = px.line(temp_df, x=x_col, y=y_col, color='city')
        
        elif plot_type == 'pie':
            if color_col:
                counts = temp_df.groupby(color_col, observed=True).size().reset_index(name='count')
                fig = px.pie(counts, values='count', names=color_col, color=color_col, **common_args)
            else:
                counts = temp_df[x_col].value_counts().reset_index()
                counts.columns = ['Category', 'Count']
                fig = px.pie(counts, values='Count', names='Category')
            
        elif plot_type == 'box':
            if y_col:
                fig = px.box(temp_df, x=x_col, y=y_col, color=color_col, **common_args)
            else:
                fig = px.box(temp_df, x=color_col, y=x_col, color=color_col, **common_args)
            
            # Vérif numérique
            val_col = y_col if y_col else x_col
            if not is_numeric_or_datetime(temp_df[val_col]):
                 return jsonify({'error': "La colonne de valeur doit être numérique."}), 400
        
        elif plot_type == 'distribution_histogram':
            fig = px.histogram(temp_df, x=x_col, color=color_col, marginal='box', **common_args)
        
        else:
            return jsonify({'error': 'Type de graphique non pris en charge.'}), 400
            
        # Mise en page légende
        if color_col:
            fig.update_layout(
                legend_title_text=color_col.replace('_category', '').replace('_', ' '),
                legend={'traceorder': 'normal'}
            )
            
        # -------------------------------------------------------------------
        # Activation du slider temporel (Uniquement si 'line' ou 'bar' activé plus haut)
        # -------------------------------------------------------------------
        if is_time_series:
            fig.update_layout(
                xaxis=dict(
                    type="date", # Force l'axe X en mode DATE (gère les trous temporels)
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1h", step="hour", stepmode="backward"),
                            dict(count=1, label="1j", step="day", stepmode="backward"),
                            dict(step="all", label="Tout")
                        ])
                    ),
                    rangeslider=dict(visible=True)
                )
            )
        
        return jsonify({'plot_json': fig.to_json()})

    except Exception as e:
        app.logger.error(f"Erreur plot: {str(e)}")
        return jsonify({'error': f"Erreur technique : {str(e)}"}), 500
# ==============================================================================
# 🎨 PAGE PREDICTION
# ==============================================================================
@app.route('/prediction-modeling')
def prediction_modeling():
    return render_template('prediction_modeling.html')

@app.route('/get_correlation_matrix', methods=['GET'])
def get_correlation_matrix():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400
    
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        return jsonify({'error': 'Le jeu de données ne contient pas de colonnes numériques pour la corrélation.'}), 400

    corr_matrix = numeric_df.corr().round(2)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        text=corr_matrix.values,       
        texttemplate='%{text}',       
        textfont={"size": 12}          
    ))

    fig.update_layout(
        title='Matrice de Corrélation',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    return jsonify({'plot_json': fig.to_json()})

@app.route('/get_data_columns', methods=['GET'])
def get_data_columns():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400
    
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return jsonify({'columns': numeric_cols})



@app.route('/train_predict', methods=['POST'])
def train_predict():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400

    time_col = session.get('time_col')
    if not time_col or time_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        return jsonify({'error': 'Colonne de temps introuvable ou de format incorrect.'}), 400

    # Gérer les NaT dans la colonne de temps
    df[time_col] = df[time_col].ffill().bfill()
    if df[time_col].isna().any():
        return jsonify({'error': 'La colonne de temps contient toujours des valeurs invalides.'}), 400

    req_data = request.json
    target_col = req_data.get('targetCol')
    feature_cols = req_data.get('featureCols')

    if not target_col or not feature_cols:
        return jsonify({'error': 'Veuillez sélectionner la variable cible et les variables explicatives.'}), 400

    # Vérification des colonnes
    for col in [target_col] + feature_cols:
        if col not in df.columns:
            return jsonify({'error': f"La colonne '{col}' est introuvable."}), 400
            
    df = df.copy()

    # ==============================================================================
    # TRAITEMENT DES VALEURS ABERRANTES (OUTLIERS) - IQR
    # ==============================================================================
    cols_to_process = [target_col] + feature_cols
    for col in cols_to_process:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            median_value = df[col].median()
            
            outlier_indexes = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            if not outlier_indexes.empty:
                df.loc[outlier_indexes, col] = median_value

    # ==============================================================================
    # INGÉNIERIE DES FONCTIONNALITÉS (FEATURE ENGINEERING)
    # ==============================================================================
    df['heure'] = df[time_col].dt.hour
    df['jour_semaine'] = df[time_col].dt.dayofweek
    df['mois'] = df[time_col].dt.month
    df['jour'] = df[time_col].dt.day
    df['minute'] = df[time_col].dt.minute
    
    df.sort_values(by=time_col, inplace=True)

    # Création des variables de "lag" (décalage)
    df[f'{target_col}_lag1'] = df[target_col].shift(1)
    for col in feature_cols:
        df[f'{col}_lag1'] = df[col].shift(1)

    # Remplissage des NaN générés par le lag et autres manques
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)

    if df.empty:
        return jsonify({'error': 'Jeu de données trop petit après traitement.'}), 400

    # Liste initiale de toutes les features potentielles
    potential_features = [f'{target_col}_lag1'] + [f'{col}_lag1' for col in feature_cols] + ['heure', 'jour_semaine', 'mois', 'jour', 'minute']
    potential_features = [col for col in potential_features if col in df.columns]

    # ==============================================================================
    # SÉLECTION DES VARIABLES (SANS FILTRE DE CORRÉLATION)
    # ==============================================================================
    # On utilise directement toutes les features potentielles sans filtrage
    features_final = potential_features

    # Préparation X et y
    X = df[features_final]
    y = df[target_col]

    # Vérification taille minimale
    if len(df) < 2:
        return jsonify({'error': 'Pas assez de données pour entraîner un modèle.'}), 400

    # Split Train/Test (80/20)
    split_point = int(len(df) * 0.8)
    if split_point == 0: split_point = 1 # Sécurité pour très petits datasets
    
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    if X_train.empty or X_test.empty:
        return jsonify({'error': 'Données insuffisantes pour le split Train/Test.'}), 400

    # ==============================================================================
    # MODÉLISATION (AVEC RÉGRESSION LINÉAIRE)
    # ==============================================================================
    models = {
        "Linear Regression": LinearRegression(), 
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Sauvegarde du modèle
            model_filename = f'modele_{name.replace(" ", "_")}.joblib'
            joblib.dump(model, os.path.join(app.config['OUTPUT_FOLDER'], model_filename))

            results[name] = {
                'mse': mse,
                'r2': r2,
                'mae': mae
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    # ==============================================================================
    # PRÉDICTION ITÉRATIVE (FUTURE)
    # ==============================================================================
    def predict_next_hours(model_name, num_steps=18):
        model_path = os.path.join(app.config['OUTPUT_FOLDER'], f'modele_{model_name.replace(" ", "_")}.joblib')
        if not os.path.exists(model_path):
            return []
            
        model = joblib.load(model_path)
        last_row = df.tail(1).copy()
        predictions = []
        current_data = last_row.iloc[0].to_dict()

        for _ in range(num_steps):
            # Mise à jour des lags
            current_data[f'{target_col}_lag1'] = current_data[target_col]
            for col in feature_cols:
                current_data[f'{col}_lag1'] = current_data[col]
            
            # Avance temporelle (10 minutes)
            current_data[time_col] += timedelta(minutes=10)
            
            # Mise à jour features temporelles
            current_data['heure'] = current_data[time_col].hour
            current_data['minute'] = current_data[time_col].minute
            current_data['jour_semaine'] = current_data[time_col].dayofweek
            current_data['jour'] = current_data[time_col].day
            current_data['mois'] = current_data[time_col].month

            
            try:
                predict_row = pd.DataFrame([current_data])
                # Filtrer pour ne garder que les colonnes utilisées lors de l'entraînement
                predict_df = predict_row[features_final] 
                
                prediction_value = model.predict(predict_df)[0]
                
                predictions.append({
                    'time': current_data[time_col].strftime('%Y-%m-%d %H:%M'),
                    'value': prediction_value
                })
                
                # Mise à jour de la cible pour la prochaine boucle
                current_data[target_col] = prediction_value
            except Exception as e:
                print(f"Erreur prédiction itérative: {e}")
                break

        return predictions

    # Génération des prédictions pour les 3 modèles
    predictions_lr = predict_next_hours("Linear Regression")
    predictions_rf = predict_next_hours("Random Forest")
    predictions_gb = predict_next_hours("Gradient Boosting")

    return jsonify({
        'message': 'Modèles entraînés et prédictions générées avec succès.',
        'features_selected': features_final, # Info utile pour le front-end
        'results': results,
        'predictions_lr': predictions_lr,
        'predictions_rf': predictions_rf,
        'predictions_gb': predictions_gb
    })
@app.route('/get_comparison_columns', methods=['GET'])
def get_comparison_columns():
    """
    Route pour obtenir les colonnes textuelles (pour la ville) et numériques
    (pour les données météo) afin de peupler les menus déroulants.
    """
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # On suppose que les colonnes de type 'object' ou 'category' peuvent contenir le nom de la ville
    text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return jsonify({
        'numeric_columns': numeric_cols,
        'text_columns': text_cols
    })


# ------------------------------------------------------------------------------
# 1. ROUTE POUR AFFICHER LA PAGE HTML (MÉTHODE GET)
# ------------------------------------------------------------------------------

@app.route('/comparaison')
def comparaison_page():
    """Affiche la page de comparaison météo."""
    return render_template('comparaison_page.html')


# ------------------------------------------------------------------------------
# 2. ROUTE POUR TRAITER LES DONNÉES ET GÉNÉRER LES GRAPHIQUES (MÉTHODE POST)
# ------------------------------------------------------------------------------
@app.route('/compare_with_meteo', methods=['POST']) 
def compare_with_meteo():
    """
    Compare les données du capteur avec les données de Meteostat.
    La ville est fournie directement par l'utilisateur (via input ou select).
    """
    df = get_df_from_session()
    time_col = session.get('time_col')

    # Validation initiale
    if df is None or not time_col:
        return jsonify({'error': 'Données ou colonne de temps non trouvées en session.'}), 400

    # Vérifie si la requête contient bien du JSON
    if not request.is_json:
        return jsonify({'error': "La requête doit être au format JSON."}), 415

    req_data = request.get_json()
    temp_col = req_data.get('temp_col')
    humidity_col = req_data.get('humidity_col')
    city_name = req_data.get('city_name')

    if not all([temp_col, humidity_col, city_name]):
        return jsonify({'error': "Veuillez sélectionner les colonnes et choisir une ville."}), 400

    # Préparation des données utilisateur
    try:
        df_prepared = df.copy()
        df_prepared[time_col] = pd.to_datetime(df_prepared[time_col], dayfirst=True) # dayfirst aide souvent
        
        start_date = df_prepared[time_col].min()
        end_date = df_prepared[time_col].max()
    except Exception as e:
        return jsonify({'error': f"Erreur de format de date : {e}"}), 400

    # Géolocalisation de la ville
    try:
        # User-agent personnalisé pour éviter le blocage Nominatim
        geolocator = Nominatim(user_agent="pollugard-app-v3-comparison", timeout=10)
        location = geolocator.geocode(city_name)
        
        if location is None:
            return jsonify({'error': f"La ville '{city_name}' est introuvable. Vérifiez l'orthographe."}), 400
            
        meteo_point = Point(location.latitude, location.longitude)
        
    except Exception as e:
        return jsonify({'error': f"Erreur de géolocalisation (service externe) : {e}"}), 500

    # Récupération des données Meteostat
    try:
        meteo_data = Hourly(meteo_point, start_date, end_date)
        meteo_df = meteo_data.fetch()
        
        if meteo_df.empty:
            return jsonify({'message': f"Aucune donnée météo officielle trouvée pour {city_name} sur cette période."}), 200
            
    except Exception as e:
        return jsonify({'error': f"Erreur lors de la récupération Meteostat : {e}"}), 500

    # Fusion et alignement des données
    try:
        local_timezone = 'Europe/Paris'
        
        # 1. Gestion Timezone Meteostat (toujours en UTC)
        if meteo_df.index.tz is None:
            meteo_df.index = meteo_df.index.tz_localize('UTC')
        meteo_df.index = meteo_df.index.tz_convert(local_timezone)

        # 2. Gestion Timezone Données Capteur
        # Si les données n'ont pas de timezone, on suppose qu'elles sont déjà en local (Paris)
        if df_prepared[time_col].dt.tz is None:
            df_prepared[time_col] = df_prepared[time_col].dt.tz_localize(local_timezone, ambiguous='infer')
        else:
            df_prepared[time_col] = df_prepared[time_col].dt.tz_convert(local_timezone)
            
        df_indexed = df_prepared.set_index(time_col)
        
        # Rééchantillonnage horaire pour aligner avec Meteostat
        df_resampled = df_indexed.resample('H').mean(numeric_only=True) 

        # Fusion (Inner join pour ne garder que les moments communs)
        comparison_df = pd.merge(df_resampled, meteo_df, left_index=True, right_index=True, how='inner')

        if comparison_df.empty:
            return jsonify({'message': 'Les périodes de temps ne correspondent pas (aucun chevauchement).'}), 200

    except Exception as e:
        return jsonify({'error': f"Erreur lors du traitement temporel : {e}"}), 500

    # Création des graphiques Plotly
    plots = {}

    # Graphique Température
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[temp_col], name='Température Capteur', mode='lines', line=dict(color='blue')))
    fig_temp.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['temp'], name='Température Météo France/Station', mode='lines', line=dict(color='orange')))
    fig_temp.update_layout(
        title=f'Comparaison Température - {city_name}',
        xaxis_title='Date',
        yaxis_title='Température (°C)',
        hovermode="x unified"
    )
    plots['temperature'] = fig_temp.to_json()

    # Graphique Humidité
    fig_humidity = go.Figure()
    fig_humidity.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[humidity_col], name='Humidité Capteur', mode='lines', line=dict(color='blue')))
    fig_humidity.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['rhum'], name='Humidité Météo France/Station', mode='lines', line=dict(color='orange')))
    fig_humidity.update_layout(
        title=f'Comparaison Humidité - {city_name}',
        xaxis_title='Date',
        yaxis_title='Humidité Relative (%)',
        hovermode="x unified"
    )
    plots['humidity'] = fig_humidity.to_json()

    return jsonify({
        'status': 'success',
        'city': city_name,
        'plots': plots
    })

# ==============================================================================
# 🧠 MOTEUR D'ANALYSE INTELLIGENT (CORRIGÉ)
# ==============================================================================
# ==============================================================================
# 1. BASE DE CONNAISSANCE ENRICHIE (Agriculture, Industrie, BTP...)
# ==============================================================================
POLLUTANT_KNOWLEDGE = {
    'PM': {
        'desc': "Particules en suspension (Poussières, fumées).",
        'hausse': "Trafic, Chauffage bois, Industrie, Agriculture, Chantiers.",
        'baisse': "Pluie, Vent fort, Arrêt activité."
    },
    'PM10': {
        'desc': "Particules < 10µm (Irritantes).",
        'hausse': "Chauffage, Usure routes/freins, Épandages agricoles, Chantiers BTP.",
        'baisse': "Lessivage (Pluie), Dispersion."
    },
    'PM2.5': {
        'desc': "Particules < 2.5µm (Nocives, pénètrent le sang).",
        'hausse': "Combustion (Moteurs, Chaudières, Brûlage déchets verts), Industrie.",
        'baisse': "Instabilité atmosphérique."
    },
    'COV': {
        'desc': "Composés Organiques Volatils (Chimie).",
        'hausse': "Solvants, Peintures, Nettoyage, Industrie, Trafic.",
        'baisse': "Vent, Photolyse (Soleil)."
    },
    'CO2': {
        'desc': "Dioxyde de carbone (Confinement).",
        'hausse': "Respiration humaine (salle pleine), Chaudières, Manque d'aération.",
        'baisse': "Ouverture fenêtres, Végétation (jour)."
    },
    'NOX': {
        'desc': "Oxydes d'azote (Marqueur combustion).",
        'hausse': "Moteurs Diesel, Centrales thermiques, Industrie lourde.",
        'baisse': "Réactions avec l'Ozone."
    },
    'TEMPERATURE': {'desc': "Température air.", 'hausse': "Soleil, Activité urbaine.", 'baisse': "Nuit, Vent."},
    'HUMIDITE': {'desc': "Humidité relative.", 'hausse': "Pluie, Respiration/Cuisine (intérieur).", 'baisse': "Soleil, Chauffage sec."}
}



# Mapping pour relier les noms de colonnes bizarres aux clés ci-dessus
VAR_MAPPING = {
    'temp': 'TEMPERATURE', 't°': 'TEMPERATURE',
    'hum': 'HUMIDITE', 'rh': 'HUMIDITE',
    'pm10': 'PM10', 'pm2.5': 'PM2.5', 'pm1': 'PM2.5', 'pm2_5': 'PM2.5',
    'cov': 'COV', 'voc': 'COV',
    'co2': 'CO2',
    'nox': 'NOX',
    'lum': 'LUMIERE', 'lux': 'LUMIERE'
}

def get_time_context(date):
    h = date.hour
    day = date.dayofweek
    is_weekend = day >= 5
    period = "Nuit"
    if 6 <= h <= 9: period = "Pointe Matin"
    elif 10 <= h <= 16: period = "Journée"
    elif 17 <= h <= 20: period = "Pointe Soir"
    season = "Automne"
    m = date.month
    if m in [12, 1, 2]: season = "Hiver"
    elif m in [3, 4, 5]: season = "Printemps"
    elif m in [6, 7, 8]: season = "Été"
    return period, season, is_weekend

# ==============================================================================
# 2. LOGIQUE D'ANALYSE AVANCÉE (Incluant Activités Humaines Diverses)
# ==============================================================================
def generate_explanation(knowledge_key, value, context):
    """
    Génère l'explication en croisant Polluant + Valeur + Temps + Activité Humaine
    """
    if not knowledge_key: return "Analyse non disponible."
    
    period, season, is_weekend = context
    causes = []
    key = knowledge_key.upper()
    
    # --- LOGIQUE PARTICULES (PM10, PM2.5) ---
    if 'PM' in key:
        # 1. Trafic (Classique)
        if period in ['Pointe Matin', 'Pointe Soir'] and not is_weekend: 
            causes.append("trafic routier pendulaire")
        
        # 2. Chauffage (Hiver/Soir/Nuit)
        if season in ['Hiver', 'Automne'] and period in ['Pointe Soir', 'Nuit']: 
            causes.append("chauffage résidentiel (bois/fioul)")
        
        # 3. Agriculture (Printemps/Automne + Journée) -> Souvent oublié !
        if season in ['Printemps', 'Automne'] and period == 'Journée':
            causes.append("épandages agricoles ou labours (poussières)")

        # 4. Industrie / Chantiers (Semaine + Journée)
        if not is_weekend and period == 'Journée':
            causes.append("activité industrielle ou chantiers BTP proches")
            
        # 5. Météo (Hiver + Matin)
        if season == 'Hiver' and period == 'Pointe Matin': 
            causes.append("inversion thermique (polluants piégés)")

    # --- LOGIQUE COV (Chimie) ---
    elif 'COV' in key or 'VOC' in key:
        # 1. Trafic
        if period in ['Pointe Matin', 'Pointe Soir']: 
            causes.append("gaz d'échappement")
        
        # 2. Activité Domestique / Bricolage (Week-end ou Journée)
        if (is_weekend and period == 'Journée') or (period == 'Journée'):
            causes.append("usage de solvants, peintures ou produits ménagers")
            
        # 3. Industrie (Semaine)
        if not is_weekend and period == 'Journée':
            causes.append("émissions industrielles (usines, pressings)")

        # 4. Naturel (Été)
        if season == 'Été' and period == 'Journée': 
            causes.append("évaporation thermique (végétation/carburants)")

    # --- LOGIQUE CO2 (Confinement) ---
    elif 'CO2' in key:
        # 1. Occupation Humaine
        if value > 1000:
            causes.append("forte occupation humaine (réunions, classe, foule)")
        
        # 2. Manque d'aération
        if season == 'Hiver': 
            causes.append("confinement (fenêtres fermées pour chauffer)")
            
        # 3. Combustion interne
        if value > 600 and period == 'Journée':
             causes.append("respiration ou cuisine (si capteur intérieur)")

    # --- LOGIQUE NOX (Industrie/Route) ---
    elif 'NOX' in key:
        if period in ['Pointe Matin', 'Pointe Soir']:
            causes.append("trafic routier (véhicules diesel)")
        if not is_weekend and period == 'Journée':
            causes.append("activités industrielles ou logistiques (camions)")

    # --- LOGIQUE TEMPÉRATURE ---
    elif 'TEMPERATURE' in key:
        if value < 0: causes.append("gel hivernal")
        elif value > 30: causes.append("canicule")
        elif period == 'Journée': causes.append("ensoleillement")
        elif period == 'Nuit': causes.append("refroidissement nocturne")

    # --- LOGIQUE HUMIDITÉ ---
    elif 'HUMIDITE' in key:
        if value > 90: causes.append("temps pluvieux ou brouillard")
        elif period == 'Nuit' and value > 70: causes.append("rosée nocturne")
        elif period == 'Journée' and value < 40: causes.append("air sec / chauffage actif")

    if not causes: 
        return "Source locale indéterminée ou pollution de fond."

    return f"Facteurs possibles : {', '.join(causes)}."
@app.route('/analyze_peaks', methods=['POST'])
def analyze_peaks():
    try:
        df = get_df_from_session()
        if df is None: return jsonify({'error': 'Données non trouvées'}), 400
        
        req = request.json
        raw_pollutant = req.get('pollutant', '') # ex: "Température (°C)" ou "PM10"
        
        # 1. Trouver la colonne de données dans le DataFrame
        data_col = next((c for c in df.columns if raw_pollutant.lower() in c.lower()), None)
        if not data_col: return jsonify({'error': f"Colonne '{raw_pollutant}' introuvable."}), 400

        # 2. Identifier la clé de connaissance (Normalisation)
        # On cherche des bouts de mots (ex: "temp" dans "Température")
        knowledge_key = None
        normalized_name = "Variable Inconnue"
        
        for snippet, key in VAR_MAPPING.items():
            if snippet in raw_pollutant.lower() or snippet in data_col.lower():
                knowledge_key = key
                normalized_name = key
                break
        
        # 3. Récupérer les infos encyclopédiques
        if knowledge_key:
            knowledge = POLLUTANT_KNOWLEDGE.get(knowledge_key)
        else:
            # Cas par défaut pour variable inconnue
            knowledge = {
                'desc': "Aucune définition encyclopédique disponible pour cette variable.",
                'hausse': "Non défini.",
                'baisse': "Non défini."
            }
            normalized_name = raw_pollutant # On garde le nom original

        # 4. Identification du Groupe (Ville)
        group_col = next((c for c in df.columns if c.lower() in ['city', 'ville', 'commune', 'nom', 'source']), None)
        
        analysis_results = {}

        def analyze_subset(sub_df, group_name):
            valid_df = sub_df.dropna(subset=[data_col])
            if valid_df.empty: return None
            
            # Stats locales
            local_mean = valid_df[data_col].mean()
            local_max = valid_df[data_col].max()
            local_min = valid_df[data_col].min()
            
            # Pics (Top 3 Max)
            peaks_df = valid_df.nlargest(3, data_col)
            peaks_list = []
            for _, row in peaks_df.iterrows():
                ctx = get_time_context(row['temps'])
                peaks_list.append({
                    'time': row['temps'].strftime('%d/%m %H:%M'),
                    'value': round(row[data_col], 2),
                    'explanation': generate_explanation(knowledge_key, row[data_col], ctx)
                })

            # Creux (Top 3 Min)
            troughs_df = valid_df.nsmallest(3, data_col)
            troughs_list = []
            for _, row in troughs_df.iterrows():
                troughs_list.append({
                    'time': row['temps'].strftime('%d/%m %H:%M'),
                    'value': round(row[data_col], 2)
                })
            
            return {
                'stats': {
                    'avg': round(local_mean, 2),
                    'max': round(local_max, 2),
                    'min': round(local_min, 2)
                },
                'peaks': peaks_list,
                'troughs': troughs_list
            }

        # Exécution
        if group_col:
            groups = df[group_col].dropna().unique()
            for g in groups:
                res = analyze_subset(df[df[group_col] == g], str(g))
                if res: analysis_results[str(g)] = res
        else:
            res = analyze_subset(df, "Analyse Globale")
            if res: analysis_results["Global"] = res

        return jsonify({
            'pollutant': normalized_name, # Nom propre (ex: TEMPERATURE)
            'knowledge': knowledge,
            'analysis': analysis_results
        })

    except Exception as e:
        print(f"ERREUR: {e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
