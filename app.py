import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify, session, send_file
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
import io
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

def get_authenticated_session(username, password):
    session_http = requests.Session()
    login_url = "https://polluguard.eurosmart.fr/api/login"
    
    try:
        response = session_http.post(login_url, json={
            "username": username,
            "password": password
        }, timeout=10)
        
        if response.status_code == 200:
            return session_http  # La session contient maintenant le cookie de connexion
        return None
    except Exception as e:
        print(f"Erreur de connexion : {e}")
        return None
# ==============================================================================
# 📥 ROUTE D'UPLOAD
# ==============================================================================
# Route pour afficher la page de chargement (upload_data.html)
@app.route('/upload-data')
def upload_data():
    return render_template('upload_data.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    
    # --- OPTION A : FICHIER LOCAL ---
    if 'file_local' in request.files and request.files['file_local'].filename != '':
        file = request.files['file_local']
        city_name = request.form.get('city_manual', 'Inconnu').strip() or "Local_Import"
        
        try:
            filename = file.filename
            if filename.endswith('.csv'):
                # On tente de détecter le délimiteur automatiquement ou on utilise le standard
                df_temp = pd.read_csv(file, sep=None, engine='python', decimal=',')
            elif filename.endswith(('.xls', '.xlsx')):
                df_temp = pd.read_excel(file)
            else:
                return jsonify({'error': 'Format de fichier non supporté (CSV ou Excel uniquement).'}), 400

            if df_temp.empty:
                return jsonify({'error': 'Le fichier est vide.'}), 400

            # Nettoyage minimal des colonnes pour rester compatible avec tes scripts
            df_temp.columns = df_temp.columns.astype(str).str.strip().str.replace(' ', '')
            
            # Mapping des colonnes (réutilisation de ta logique existante)
            column_mapping = {
                '_HR': 'Humidité', '_Temp': 'Température', '_LUM': 'Lumière', '_VOC': 'COV',
                '_PM1': 'PM1', '_PM2': 'PM2_5', '_PM4': 'PM4', '_PM10': 'PM10',
                '_IQA': 'IQA', '_CO2': 'CO2', '_NOX': 'NOX', '_PA': 'Pression', 'Villes' : 'city' 
            }
            rename_dict = {}
            for original, new in column_mapping.items():
                for col in df_temp.columns:
                    if original.lower() in col.lower():
                        rename_dict[col] = new
                        break
            df_temp.rename(columns=rename_dict, inplace=True)

            # Gestion de la colonne temps
            df_temp, time_col = convert_to_datetime(df_temp)
            if time_col: 
                session['time_col'] = time_col
                df_temp = df_temp.sort_values(by=time_col)
            
            df_temp['city'] = city_name
            
            # Sauvegarde en session (Parquet)
            data_filename = f'{uuid.uuid4()}.parquet'
            file_path = os.path.join(app.config['DATA_FOLDER'], data_filename)
            df_temp.to_parquet(file_path)
            session['data_file'] = data_filename

            return jsonify({
                'message': 'Fichier local chargé avec succès.',
                'data_info': {
                    'rows': len(df_temp),
                    'columns': list(df_temp.columns),
                    'uploaded_files': [{'filename': filename, 'city': city_name}]
                },
                'preview': df_temp.head(10).to_html(classes='data-table table-striped table-bordered', index=False)
            })

        except Exception as e:
            return jsonify({'error': f'Erreur lors de la lecture du fichier : {str(e)}'}), 500
    
    # 1. Récupération des entrées (Identifiants + Codes)
    username = request.form.get('username')
    password = request.form.get('password')
    codes_input = request.form.get('code_exp')
    manual_city_fallback = request.form.get('city_exp', '').strip()
    
    if not username or not password:
        return jsonify({'error': 'Identifiant et mot de passe requis pour le téléchargement sécurisé.'}), 400
    
    
    if not codes_input:
        return jsonify({'error': 'Veuillez entrer au moins un code expérience.'}), 400
        
    # 2. Authentification
    auth_session = get_authenticated_session(username, password)
    if not auth_session:
        return jsonify({'error': 'Échec de la connexion Eurosmart. Vérifiez vos identifiants.'}), 401

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

    # --- ÉTAPE API : RÉCUPÉRER LA LISTE POUR GÉO ---
    api_experiments = []
    try:
        url = "https://polluguard.eurosmart.fr/get_experiments"
        token = "IUFNIFN-9z84fSION@soi-efgzerg"
        headers = {"Authorization": f"Bearer {token}"}
        response = auth_session.get(url, headers=headers, timeout=5) 
        if response.status_code == 200:
            api_experiments = response.json()
    except Exception as e:
        app.logger.warning(f"API Eurosmart warning (liste géo): {e}")

    # --- ÉTAPE : TRAITEMENT DES CODES ---
    normalized_input = codes_input.replace(';', ',').replace(' ', ',')
    code_list = [c.strip() for c in normalized_input.split(',') if c.strip()]

    for i, code in enumerate(code_list):
        if i > 0: time.sleep(1.0) # Petit délai pour ne pas surcharger le serveur

        try:
            # A. Téléchargement Sécurisé
            download_url = f"https://polluguard.eurosmart.fr/secure_download?codeExp={code}"
            response = auth_session.get(download_url)
            
            if response.status_code != 200:
                errors_log.append(f"{code}: Accès refusé ou code inexistant.")
                continue

            # Lecture du CSV depuis le flux texte reçu
            from io import StringIO
            df_temp = pd.read_csv(StringIO(response.text), sep=delimiter, decimal=decimal, low_memory=False)
            
            if df_temp.empty:
                 errors_log.append(f"{code}: Fichier vide.")
                 continue

            # B. Détection Ville (Ta logique reste la même)
            city_name = "Inconnu"
            exp_info = next((item for item in api_experiments if item.get("CodeExp") == code), None)
            
            if exp_info and 'Latitude' in exp_info and 'Longitude' in exp_info:
                try:
                    detected_city = get_city_from_coordinates(exp_info['Latitude'], exp_info['Longitude'])
                    if detected_city: city_name = detected_city
                except Exception: pass
            
            if city_name == "Inconnu" and '_' in code:
                parts = code.split('_')
                if len(parts[0]) > 2 and not parts[0].isdigit():
                    city_name = parts[0].capitalize()
            
            if city_name == "Inconnu" and manual_city_fallback:
                city_name = manual_city_fallback
            
            if city_name == "Inconnu":
                city_name = f"Exp_{code}"

            # C. Nettoyage et Mapping (Ta logique reste la même)
            df_temp.columns = df_temp.columns.astype(str).str.strip().str.replace(' ', '')
            rename_dict = {}
            for original, new in column_mapping.items():
                for col in df_temp.columns:
                    if original.lower() in col.lower() and col not in rename_dict.values():
                        rename_dict[col] = new
                        break
            if rename_dict: df_temp.rename(columns=rename_dict, inplace=True)
            
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
                
            df_temp, time_col = convert_to_datetime(df_temp)
            if time_col: session['time_col'] = time_col
            
            df_temp['city'] = city_name
            
            for col in df_temp.columns:
                if pd.api.types.is_numeric_dtype(df_temp[col]):
                    df_temp[col].fillna(df_temp[col].median(), inplace=True)

            file_info_list.append({'filename': code, 'city': city_name})
            all_data = pd.concat([all_data, df_temp], ignore_index=True)

        except Exception as e:
            app.logger.error(f"Erreur sur le code {code}: {e}")
            errors_log.append(f"{code}: Erreur interne ({str(e)}).")
            
    # --- 3. SAUVEGARDE ET APERÇU ---
    if all_data.empty:  
        msg = "Échec du chargement."
        if errors_log: msg += " Détails : " + " | ".join(errors_log)
        return jsonify({'error': msg}), 400
            
    data_filename = f'{uuid.uuid4()}.parquet'
    file_path = os.path.join(app.config['DATA_FOLDER'], data_filename)
    all_data.to_parquet(file_path)
    session['data_file'] = data_filename
    
    try:
        preview_df = all_data.groupby('city', sort=False).head(5).reset_index(drop=True)
        preview_html = preview_df.to_html(classes='data-table table-striped table-bordered', index=False)
    except Exception:
        preview_html = all_data.head(10).to_html(classes='data-table table-striped table-bordered')

    info = {        
        'rows': len(all_data),
        'columns': list(all_data.columns),
        'cols_info': all_data.dtypes.astype(str).to_dict(),
        'uploaded_files': file_info_list
    }
    
    msg = 'Données téléchargées et chargées avec succès.'
    if errors_log: msg += " (Attention: " + ", ".join(errors_log) + ")"
    
    return jsonify({'message': msg, 'data_info': info, 'preview': preview_html})

@app.route('/download_data')
def download_data():
    """
    Permet de télécharger les données actuelles (chargées en session) sous format CSV.
    """
    df = get_df_from_session()
    if df is None:
        return "Aucune donnée disponible. Veuillez d'abord charger des fichiers.", 404
    
    try:
        # Création d'un buffer mémoire
        buffer = io.BytesIO()
        
        # Export en CSV (Format Excel Français : séparateur point-virgule, encodage utf-8-sig)
        df.to_csv(buffer, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name='donnees_eurosmart_completes.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        app.logger.error(f"Erreur lors du téléchargement : {e}")
        return f"Erreur serveur : {e}", 500  
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


# ==============================================================================
# 🧠 MAPPING INTELLIGENT (Colonne FR -> Logique EN)
# ==============================================================================
# Permet de deviner quel jeu de seuils appliquer selon le nom de la colonne
AUTO_CATEGORY_MAP = {
    'Température': 'temperature',
    'Humidité': 'humidity',
    'COV': 'cov',
    'IQA': 'iqa',
    'PM': 'pm',       # Matchera PM1, PM2_5, PM10
    'CO2': 'co2',
    'NOX': 'nox',
    'Lumière': 'light'
}

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400
    
    req = request.json
    plot_type = req.get('plotType')
    x_col = req.get('xCol')
    y_col = req.get('yCol')
    color_col = req.get('colorCol')
    city = req.get('city')
    
    # Nettoyage entrées vides
    if x_col == "": x_col = None
    if y_col == "": y_col = None
    if color_col == "": color_col = None

    temp_df = df.copy()

    # 1. FILTRAGE VILLE
    if city and 'city' in temp_df.columns:
        temp_df = temp_df[temp_df['city'] == city]
        if temp_df.empty: return jsonify({'error': f"Pas de données pour {city}"}), 400

    # 2. GESTION INTELLIGENTE DES CATÉGORIES / COULEURS
    # Le JS envoie "Température" dans color_col. On doit deviner que c'est le type 'temperature'.
    category_orders = {}
    color_map = None
    
    if color_col:
        # Déduction automatique du type de seuil (ex: "Température" -> "temperature")
        found_type = None
        for fr_key, en_key in AUTO_CATEGORY_MAP.items():
            if fr_key in color_col: # Si "PM" est dans "PM2_5", on prend "pm"
                found_type = en_key
                break
        
        # Si on a trouvé un type connu, on transforme la colonne numérique en catégories
        if found_type:
            try:
                # On appelle VOTRE fonction qui crée la colonne _category
                temp_df, new_col, color_map, ordered_cats = create_color_categories(temp_df, color_col, found_type)
                
                # IMPORTANT : On remplace la colonne cible par la version catégorielle
                # Pour un Pie chart ou une coloration, on veut les catégories (Chaud/Froid), pas les chiffres.
                color_col = new_col 
                category_orders[color_col] = ordered_cats
            except Exception as e:
                app.logger.error(f"Erreur catégorisation: {e}")

    # 3. PRÉPARATION TEMPORELLE (Uniquement pour Line/Bar)
    is_time_series = False
    if plot_type in ['line', 'bar']:
        if x_col and x_col in temp_df.columns:
            temp_df[x_col] = pd.to_datetime(temp_df[x_col], errors='coerce')
            temp_df = temp_df.sort_values(x_col)
            is_time_series = True

    # 4. GÉNÉRATION DES GRAPHIQUES
    try:
        common_args = {
            'color_discrete_map': color_map,
            'category_orders': category_orders
        }

        fig = None

        if plot_type == 'scatter':
            # Nuage de points : X=Num, Y=Num, Color=Catégorie
            fig = px.scatter(temp_df, x=x_col, y=y_col, color=color_col, **common_args)

        elif plot_type == 'bar':
            # Barres : X=Temps, Y=Num, Color=Catégorie
            fig = px.bar(temp_df, x=x_col, y=y_col, color=color_col, barmode='group', **common_args)

        elif plot_type == 'line':
            # Ligne : X=Temps, Y=Num, Color=Catégorie ou Ville
            color_target = color_col if color_col else ('city' if not city and 'city' in temp_df.columns else None)
            fig = px.line(temp_df, x=x_col, y=y_col, color=color_target, **common_args)

        elif plot_type == 'pie':
            # Cible : La colonne de couleur (ex: Température_category) ou la Ville
            target = color_col if color_col else 'city'
            
            # Agrégation
            counts = temp_df[target].value_counts().reset_index()
            counts.columns = [target, 'count']
            
            # Si on a une color_map (ex: Température), on force les couleurs
            if color_map:
                fig = px.pie(counts, values='count', names=target, color=target, **common_args)
            else:
                # Sinon (ex: Ville), on laisse Plotly choisir les couleurs
                fig = px.pie(counts, values='count', names=target)
            
            fig.update_traces(textinfo='percent+label')

        elif plot_type == 'box':
    
            if color_col:
                fig = px.box(temp_df, x=color_col, y=y_col, color=color_col, **common_args)
            else:
                temp_df[' _'] = 'Distribution Globale'
                fig = px.box(temp_df, x=' _', y=y_col, **common_args)
                
            fig.update_layout(xaxis_title="")

        elif plot_type == 'distribution_histogram':
            
            target_val = x_col if x_col else y_col
            fig = px.histogram(temp_df, x=target_val, color=color_col, marginal="box", **common_args)

        # Mise en page finale
        if fig:
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
                legend_title_text="" 
            )
           
            if is_time_series:
                fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
            
            return jsonify({'plot_json': fig.to_json()})
        else:
             return jsonify({'error': "Type de graphique inconnu"}), 400

    except Exception as e:
        app.logger.error(f"Plot Error: {e}")
        return jsonify({'error': f"Erreur génération : {str(e)}"}), 500

# ==============================================================================
# 🎨 PAGE PREDICTION
# ==============================================================================

import pandas as pd
import numpy as np
import requests
import io
import os
import time
import uuid
import math
import re
import glob
from datetime import timedelta, datetime
from flask import Flask, render_template, request, jsonify, session, send_file
import plotly.express as px
import plotly.graph_objects as go
import joblib

# --- IMPORTS ML AVANCÉS ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Gestion des librairies optionnelles (si non installées)
try:
    from xgboost import XGBRegressor
except ImportError: XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError: LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError: CatBoostRegressor = None

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

# ==============================================================================
# 🧠 MOTEUR DE PRÉDICTION AVANCÉ (ROUTE MISE À JOUR)
# ==============================================================================
@app.route('/train_predict', methods=['POST'])
def train_predict():
    df = get_df_from_session()
    if df is None:
        return jsonify({'error': 'Aucune donnée chargée en session.'}), 400

    time_col = session.get('time_col')
    if not time_col or time_col not in df.columns:
        return jsonify({'error': 'Colonne de temps introuvable.'}), 400

    # Nettoyage Temps
    df[time_col] = df[time_col].ffill().bfill()
    
    req_data = request.json
    target_col = req_data.get('targetCol')
    feature_cols = req_data.get('featureCols')
    selected_model_key = req_data.get('selectedModel', 'all') # Nouveau paramètre

    if not target_col or not feature_cols:
        return jsonify({'error': 'Sélectionnez la cible et les variables.'}), 400

    df = df.copy()

    # 1. OUTLIERS (IQR)
    for col in [target_col] + feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            median_val = df[col].median()
            df.loc[(df[col] < lower) | (df[col] > upper), col] = median_val

    # 2. FEATURE ENGINEERING (Temporel)
    df['heure'] = df[time_col].dt.hour
    df['jour_semaine'] = df[time_col].dt.dayofweek
    df['mois'] = df[time_col].dt.month
    
    # Tri Chronologique IMPÉRATIF
    df.sort_values(by=time_col, inplace=True)

    # Création des Lags (Mémoire du passé)
    # Pour éviter la triche, on utilise t-1 pour prédire t
    df[f'{target_col}_lag1'] = df[target_col].shift(1)
    for col in feature_cols:
        df[f'{col}_lag1'] = df[col].shift(1)

    df.dropna(inplace=True)

    if df.empty: return jsonify({'error': 'Données insuffisantes après traitement.'}), 400

    # Définition des Features Finales
    features_final = [f'{target_col}_lag1'] + [f'{col}_lag1' for col in feature_cols] + ['heure', 'jour_semaine', 'mois']
    features_final = [c for c in features_final if c in df.columns]

    X = df[features_final]
    y = df[target_col]

    # SCALING (Important pour la convergence de certains modèles)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # On garde le scaler en mémoire (idéalement faudrait le sauvegarder avec joblib)
    
    # Split Chronologique (Pas de shuffle pour les séries temporelles !)
    split = int(len(df) * 0.85)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 3. DÉFINITION DE L'ARSENAL DE MODÈLES
    available_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Ajout conditionnel des modèles avancés
    if XGBRegressor:
        available_models["XGBoost"] = XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
    if LGBMRegressor:
        available_models["LightGBM"] = LGBMRegressor(n_estimators=100, verbosity=-1, random_state=42)
    if CatBoostRegressor:
        available_models["CatBoost"] = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)

    # Filtrage selon le choix utilisateur
    models_to_train = {}
    if selected_model_key == 'all':
        models_to_train = available_models
    elif selected_model_key in available_models:
        models_to_train = {selected_model_key: available_models[selected_model_key]}
    else:
        # Fallback
        models_to_train = {"Linear Regression": available_models["Linear Regression"]}

    results = {}
    predictions_data = {}

    # 4. ENTRAÎNEMENT & SAUVEGARDE
    for name, model in models_to_train.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Sauvegarde
            model_filename = f'modele_{name.replace(" ", "_")}.joblib'
            joblib.dump(model, os.path.join(app.config['OUTPUT_FOLDER'], model_filename))
            
            results[name] = {'mse': mse, 'r2': r2, 'mae': mae}
        except Exception as e:
            results[name] = {'error': str(e)}

    # 5. PRÉDICTION FUTURE (BOUCLE 24H)
    def predict_future_24h(model_name):
        path = os.path.join(app.config['OUTPUT_FOLDER'], f'modele_{model_name.replace(" ", "_")}.joblib')
        if not os.path.exists(path): return []
        
        model = joblib.load(path)
        
        # On part de la dernière ligne réelle connue
        last_row = df.tail(1).copy()
        current_data_dict = last_row.iloc[0].to_dict()
        
        preds = []
        
        # Calcul du nombre de pas pour 24h (si intervalle = 10 min, steps = 144)
        # On tente de déduire la fréquence, sinon défaut 10 min
        try:
            freq_mins = (df[time_col].iloc[-1] - df[time_col].iloc[-2]).total_seconds() / 60
            if freq_mins <= 0: freq_mins = 10
        except:
            freq_mins = 10
            
        steps_24h = int((24 * 60) / freq_mins)
        
        current_time = current_data_dict[time_col]

        for _ in range(steps_24h):
            # Mise à jour des Lags avec la valeur PRÉCÉDENTE (Réelle ou Prédite)
            # C'est ici qu'on évite la triche : on utilise la prédiction t-1 pour prédire t
            current_data_dict[f'{target_col}_lag1'] = current_data_dict[target_col]
            for c in feature_cols:
                # Hypothèse naïve : les autres variables restent constantes (ou on pourrait aussi les prédire)
                # Pour plus de robustesse, on peut utiliser leurs propres moyennes glissantes
                current_data_dict[f'{c}_lag1'] = current_data_dict[c]

            # Avance temps
            current_time += timedelta(minutes=freq_mins)
            
            # Mise à jour features temporelles
            current_data_dict['heure'] = current_time.hour
            current_data_dict['jour_semaine'] = current_time.dayofweek
            current_data_dict['mois'] = current_time.month

            # Construction vecteur X
            row_to_predict = pd.DataFrame([current_data_dict])[features_final]
            
            # Scaling (Important d'utiliser le même scaler)
            row_scaled = scaler.transform(row_to_predict)
            
            # Prédiction
            val_pred = model.predict(row_scaled)[0]
            
            preds.append({
                'time': current_time.strftime('%Y-%m-%d %H:%M'),
                'value': float(val_pred)
            })
            
            # Mise à jour de la cible actuelle pour le prochain lag
            current_data_dict[target_col] = val_pred

        return preds

    # Génération des prédictions pour tous les modèles entraînés
    for name in results.keys():
        if 'error' not in results[name]:
            predictions_data[name] = predict_future_24h(name)

    return jsonify({
        'message': 'Analyse 24H terminée.',
        'features_used': features_final,
        'results': results,
        'predictions': predictions_data # Structure unifiée : {'Linear Regression': [...], 'XGBoost': [...]}
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
# 🧠 MOTEUR D'ANALYSE INTELLIGENT (VERSION FINALE SANS UNDEFINED)
# ==============================================================================

def generate_explanation(knowledge_key, value, context, is_peak=True):
    """
    Génère une analyse pour l'extérieur.
    is_peak=True  -> Analyse du Maximum
    is_peak=False -> Analyse du Minimum (Creux)
    """
    if not knowledge_key: return "Analyse des conditions ambiantes."
    
    period, season, is_weekend = context
    causes = []
    key = knowledge_key.upper()

    # --- LOGIQUE POUR LES CREUX (MINIMA) ---
    if not is_peak:
        if 'PM' in key or 'NOX' in key or 'CO2' in key:
            causes.append("En tenant compte de la faible concentration, les causes sont probablement liées à un lessivage de l'air par la pluie ou à une dispersion efficace des polluants par des vents soutenus.")
        elif 'TEMPERATURE' in key:
            causes.append("En tenant compte du refroidissement, les causes sont probablement liées à l'absence d'ensoleillement et à la déperdition thermique du sol vers l'atmosphère (nuit claire).")
        elif 'HUMIDITE' in key:
            causes.append("En tenant compte de l'air sec, les causes sont probablement liées à l'arrivée d'une masse d'air continentale ou à un réchauffement rapide du sol qui dissipe l'humidité de surface.")
        else:
            causes.append("Les conditions météorologiques actuelles favorisent une stabilisation des niveaux aux valeurs les plus basses enregistrées.")
        return f"Analyse : {', '.join(causes)}"

    # --- LOGIQUE POUR LES PICS (MAXIMA) ---
    if 'PM' in key:
        if period == "Pointe Matin":
            if season == "Hiver":
                causes.append("En tenant compte de l'inversion thermique, les causes sont probablement liées au piégeage des particules de combustion au sol par une couche d'air froid.")
            else:
                causes.append("En tenant compte de la reprise des flux, les causes sont probablement liées aux émissions de freinage et d'échappement du trafic routier.")
        elif period in ["Pointe Soir", "Nuit"] and season in ["Hiver", "Automne"]:
            causes.append("En tenant compte du froid nocturne, les causes sont probablement liées à l'intensification du chauffage résidentiel local.")

    elif 'CO2' in key:
        if period in ["Pointe Matin", "Pointe Soir"]:
            causes.append("En tenant compte du trafic, les causes sont probablement liées à la concentration des rejets de combustion des moteurs thermiques en zone urbaine.")
        elif period == "Nuit":
            causes.append("En tenant compte du cycle végétal, les causes sont probablement liées à la respiration nocturne des plantes rejetant du CO2 en l'absence de photosynthèse.")

    elif 'TEMPERATURE' in key:
        if period == "Journée":
            causes.append("En tenant compte de l'albédo, les causes sont probablement liées au rayonnement solaire direct et à l'accumulation de chaleur sur les surfaces minérales.")
        else:
            causes.append("En tenant compte du contexte temporel, les causes sont probablement liées à une stagnation d'une masse d'air chaud ou à une inertie thermique locale.")

    elif 'HUMIDITE' in key:
        if season == "Hiver" and period == "Pointe Matin" and value > 85:
            causes.append("En tenant compte du froid matinal, les causes sont probablement liées à la saturation de l'air atteignant son point de rosée (formation de brouillard ou givre).")
        elif season == "Été" and value > 70:
            causes.append("En tenant compte de la chaleur, les causes sont probablement liées à une forte évapotranspiration ou à une instabilité pré-orageuse.")
        elif value > 85:
            causes.append("En tenant compte du taux élevé, les causes sont probablement liées à des précipitations récentes ou à un refroidissement rapide de l'air ambiant.")

    if not causes:
        return "Analyse : Fluctuations normales liées aux cycles météorologiques extérieurs."
    
    return f"Analyse : {', '.join(causes)}"

@app.route('/analyze_peaks', methods=['POST'])
def analyze_peaks():
    try:
        df = get_df_from_session()
        if df is None: return jsonify({'error': 'Données non trouvées'}), 400
        
        req = request.json
        raw_pollutant = req.get('pollutant', '')
        
        data_col = next((c for c in df.columns if raw_pollutant.lower() in c.lower()), None)
        if not data_col: return jsonify({'error': 'Polluant introuvable'}), 400

        # Identification de la clé pour éviter le "UNDEFINED"
        knowledge_key = None
        for snippet, key in VAR_MAPPING.items():
            if snippet in raw_pollutant.lower() or snippet in data_col.lower():
                knowledge_key = key
                break
        
        # Le nom affiché sera le knowledge_key (ex: PM10) au lieu de rien
        display_name = knowledge_key if knowledge_key else raw_pollutant

        group_col = next((c for c in df.columns if c.lower() in ['city', 'ville', 'nom']), None)
        analysis_results = {}

        def analyze_subset(sub_df):
            valid_df = sub_df.dropna(subset=[data_col])
            if valid_df.empty: return None
            
            # --- PIC (MAX) ---
            peak_row = valid_df.loc[valid_df[data_col].idxmax()]
            ctx_peak = get_time_context(peak_row['temps'])
            peak_data = {
    'time': peak_row['temps'].strftime('%d/%m %H:%M'),
    'value': round(peak_row[data_col], 2),
    'explanation': generate_explanation(knowledge_key, peak_row[data_col], ctx_peak, is_peak=True)
}

            # --- CREUX (MIN) ---
            trough_row = valid_df.loc[valid_df[data_col].idxmin()]
            ctx_trough = get_time_context(trough_row['temps'])
            trough_data = {
    'time': trough_row['temps'].strftime('%d/%m %H:%M'),
    'value': round(trough_row[data_col], 2),
    # IMPORTANT : bien mettre is_peak=False ici !
    'explanation': generate_explanation(knowledge_key, trough_row[data_col], ctx_trough, is_peak=False)
}
            
            return {
                'stats': {'avg': round(valid_df[data_col].mean(), 2), 'max': round(peak_row[data_col], 2), 'min': round(trough_row[data_col], 2)},
                'peaks': [peak_data],
                'troughs': [trough_data]
            }

        if group_col:
            for g in df[group_col].dropna().unique():
                res = analyze_subset(df[df[group_col] == g])
                if res: analysis_results[str(g)] = res
        else:
            res = analyze_subset(df)
            if res: analysis_results["Global"] = res

        return jsonify({
            'pollutant': display_name,  # On envoie un nom valide ici pour le badge
            'knowledge': POLLUTANT_KNOWLEDGE.get(knowledge_key, {}),
            'analysis': analysis_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/documentation')
def documentation():
    return render_template('documentation.html')
if __name__ == '__main__':
    app.run(debug=True)
