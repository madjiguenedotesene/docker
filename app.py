import pandas as pd
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
#from google import genai
#from google.genai.errors import APIError # Correction ici: utilise genai.errors
# ----------------------------------------
#from google.generativeai import types
from datetime import datetime
from meteostat import Point, Hourly
from geopy.geocoders import Nominatim



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
    Tente de convertir les colonnes de date/heure dont le nom contient 'temps'.
    Priorise la colonne 'temps' s'il y en a une.
    Retourne le nom de la colonne de temps si elle est trouvée.
    """
    time_col_name = None
    
    
    if 'temps' in df.columns:
        col = 'temps'
        try:
            temp_df = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            if not temp_df.isna().all():
                df[col] = temp_df
                time_col_name = col
                return df, time_col_name
        except Exception as e:
            app.logger.warning(f"Impossible de convertir la colonne '{col}' en datetime: {e}")

    
    for col in df.columns:
        if 'temps' in col.lower():
            try:
                temp_df = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                if not temp_df.isna().all():
                    df[col] = temp_df
                    time_col_name = col
                    break
            except Exception as e:
                app.logger.warning(f"Impossible de convertir la colonne '{col}' en datetime: {e}")
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
# 🎨 PAGE CHARGEMENT
# ==============================================================================
@app.route('/upload-data')
def upload_data():
    return render_template('upload_data.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist('file') + request.files.getlist('file[]')
        
    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné. Veuillez choisir au moins un fichier.'}), 400
        
    delimiter = ';'    
    decimal = ','
    header = request.form.get('header_row') == 'on'
    
    file_info_list = []
    all_data = pd.DataFrame()
        
    column_mapping = {
        '_HR': 'Humidité',
        '_Temp': 'Température',
        '_LUM': 'Lumière',
        '_VOC': 'COV',
        '_PM1': 'PM1',
        '_PM2': 'PM2_5',
        '_PM4': 'PM4',
        '_PM10': 'PM10',
        '_IQA': 'IQA',
        '_CO2': 'CO2',
        '_NOX': 'NOX',
        '_PA': 'Pression'
    }
    
    for file in uploaded_files:
        if file.filename != '':
            city, month = determine_city_and_month(file.filename)
            try:
                df_temp = pd.read_csv(
                    file,
                    encoding='latin-1',
                    on_bad_lines='skip',
                    sep=delimiter,
                    decimal=decimal,
                    header=0 if header else None
                )
            except Exception as e:
                app.logger.error(f"Erreur lors de la lecture du fichier {file.filename}: {str(e)}")
                continue  

            df_temp.columns = df_temp.columns.astype(str)
            df_temp.columns = df_temp.columns.str.strip().str.replace(' ', '')
    
            rename_dict = {}
            for original, new in column_mapping.items():
                for col in df_temp.columns:
                    if original.lower() in col.lower() and col not in rename_dict.values():
                        rename_dict[col] = new
                        break
                    
            if rename_dict:
                df_temp.rename(columns=rename_dict, inplace=True)
            
            cols_to_drop = []
            time_col_found = False
            
            for col in df_temp.columns:
                lower_col = col.lower()
                if 'batterie' in lower_col:
                    cols_to_drop.append(col)
                elif 'temps' in lower_col:
                    if not time_col_found:
                        time_col_found = True
                        df_temp.rename(columns={col: 'temps'}, inplace=True)
                    else:
                        cols_to_drop.append(col)
            
            if cols_to_drop:
                df_temp = df_temp.drop(columns=cols_to_drop, errors='ignore')
                
            df_temp, time_col = convert_to_datetime(df_temp)
            if time_col:
                session['time_col'] = time_col
            
            # Ajout des colonnes 'city' et 'month' au DataFrame temporaire
            df_temp['city'] = city
            df_temp['month'] = month
            
            # Gérer les valeurs manquantes pour les colonnes numériques
            for col in df_temp.columns:
                if pd.api.types.is_numeric_dtype(df_temp[col]):
                    median_val = df_temp[col].median()
                    df_temp[col].fillna(median_val, inplace=True)
            
            
            file_info_list.append({
                'filename': file.filename,
                'city': city,
                'month': month
            })
            
            all_data = pd.concat([all_data, df_temp], ignore_index=True)
            
    if all_data.empty:  
        return jsonify({'error': 'Les fichiers téléchargés ne contiennent aucune donnée valide.'}), 400
            
    data_filename = f'{uuid.uuid4()}.parquet'
    file_path = os.path.join(app.config['DATA_FOLDER'], data_filename)
    all_data.to_parquet(file_path)
    session['data_file'] = data_filename
                
    preview_html = all_data.head().to_html(classes='data-table table-striped table-bordered')
                    
    info = {        
        'rows': len(all_data),
        'columns': list(all_data.columns),
        'cols_info': all_data.dtypes.astype(str).to_dict(),
        'uploaded_files': file_info_list
    }
    return jsonify({
        'message': 'Fichiers téléchargés et combinés avec succès.',
        'data_info': info,
        'preview': preview_html
    })
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

    if plot_type != 'line' and city:
        if 'city' not in temp_df.columns:
            return jsonify({'error': "La colonne 'city' est introuvable. Assurez-vous d'avoir téléchargé un fichier avec un nom au format 'ville-mois.csv'."}), 400
        temp_df = temp_df[temp_df['city'] == city]
        if temp_df.empty:
            return jsonify({'error': f"Aucune donnée trouvée pour la ville sélectionnée ({city})."}), 400

    # Variables pour stocker la carte de couleurs et l'ordre
    color_map = None
    category_orders = {}

    if color_col and color_type:
        try:
            # Utilisation de la fonction mise à jour qui renvoie la map et l'ordre
            temp_df, new_color_col, color_map, ordered_categories = create_color_categories(temp_df, color_col, color_type)
            color_col = new_color_col
            # Définir l'ordre pour Plotly
            category_orders[color_col] = ordered_categories
        except KeyError:
            return jsonify({'error': f"La colonne '{color_col}' est introuvable ou de type incorrect pour la catégorisation."}), 400
        except Exception as e:
            app.logger.error(f"Erreur lors de la création des catégories de couleur: {str(e)}")
            return jsonify({'error': f"Erreur lors de la création des catégories de couleur: {str(e)}"}), 500

    if x_col not in temp_df.columns:
        return jsonify({'error': f"La colonne '{x_col}' est introuvable dans le jeu de données."}), 400
    if y_col and y_col not in temp_df.columns:
        return jsonify({'error': f"La colonne '{y_col}' est introuvable dans le jeu de données."}), 400
    if color_col and color_col not in temp_df.columns:
        return jsonify({'error': f"La colonne '{color_col}' est introuvable dans le jeu de données après catégorisation."}), 400

    # NOTE: La fonction 'is_numeric_or_datetime' ne doit pas être redéfinie ici
    

    try:
        # Arguments communs pour les graphiques Plotly Express
        common_args = {
            'color_discrete_map': color_map,
            'category_orders': category_orders,
        }
        
        if plot_type == 'scatter':
            fig = px.scatter(temp_df, x=x_col, y=y_col, color=color_col, **common_args)
        
        elif plot_type == 'bar':
            fig = px.bar(temp_df, x=x_col, y=y_col, color=color_col, **common_args)
            
        elif plot_type == 'line':
            # Le graphique en ligne utilise la colonne 'city' pour la couleur par défaut
            fig = px.line(df, x=x_col, y=y_col, color='city')
        
        elif plot_type == 'pie':
            if color_col:
                # Utiliser le nom de la colonne catégorisée pour le nom et la couleur
                counts = temp_df.groupby(color_col, observed=True).size().reset_index(name='count')
                fig = px.pie(counts, values='count', names=color_col, color=color_col, **common_args)
            else:
                counts = temp_df[x_col].value_counts().reset_index()
                counts.columns = ['Category', 'Count']
                fig = px.pie(counts, values='Count', names='Category')
            
        elif plot_type == 'box':
            # Utiliser y_col si fourni, sinon utiliser x_col comme valeur 
            if y_col:
                fig = px.box(temp_df, x=x_col, y=y_col, color=color_col, **common_args)
            else:
                # Si y_col n'est pas sélectionné, on utilise color_col pour l'axe X (groupement) et x_col pour l'axe Y (valeurs)
                fig = px.box(temp_df, x=color_col, y=x_col, color=color_col, **common_args)
            
            # Vérification de la colonne numérique (valeurs)
            numeric_check = temp_df[y_col] if y_col else temp_df[x_col]
            if not is_numeric_or_datetime(numeric_check):
                return jsonify({'error': "La colonne utilisée pour les valeurs (axe Y ou axe X si Y est vide) doit être numérique pour un graphique en boîte à moustaches."}), 400
        
        elif plot_type == 'distribution_histogram':
            fig = px.histogram(temp_df, x=x_col, color=color_col, marginal='box', **common_args)
        
        else:
            return jsonify({'error': 'Type de graphique non pris en charge.'}), 400
            
        # Ajustement de la mise en page de la légende si la couleur est catégorisée
        if color_col:
            fig.update_layout(
                legend_title_text=color_col.replace('_category', '').replace('_', ' '),
                legend={'traceorder': 'normal'}
            )
            
        # -------------------------------------------------------------------
        # AJOUT : Activation de la barre de zoom pour les séries temporelles
        # -------------------------------------------------------------------
        # Condition pour vérifier si l'axe X est de type date/heure et que c'est un graphique pertinent 
        if is_numeric_or_datetime(temp_df[x_col]) and not pd.api.types.is_numeric_dtype(temp_df[x_col]) and plot_type in ['line', 'bar']:
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1h", step="hour", stepmode="backward"),
                            dict(count=1, label="1j", step="day", stepmode="backward"),
                            dict(count=7, label="1s", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(step="all", label="Tout")
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                    type="date" 
                )
            )
        
            
        return jsonify({
            'plot_json': fig.to_json()
        })

    except Exception as e:
        app.logger.error(f"Erreur lors de la génération du graphique: {str(e)}")
        return jsonify({'error': f"Erreur lors de la génération du graphique. Veuillez vérifier les colonnes sélectionnées et le type de graphique : {str(e)}"}), 500
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
        hoverongaps=False
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
        return jsonify({'error': 'Colonne de temps introuvable ou de format incorrect. Assurez-vous que le nom de la colonne contient "temps" et que les données sont valides.'}), 400

    # Gérer les NaT dans la colonne de temps avant toute opération
    df[time_col] = df[time_col].ffill().bfill()
    
    if df[time_col].isna().any():
        return jsonify({'error': 'Après le remplissage des valeurs manquantes, la colonne de temps contient toujours des valeurs invalides. Veuillez vérifier vos données.'}), 400

    req_data = request.json
    target_col = req_data.get('targetCol')
    feature_cols = req_data.get('featureCols')

    if not target_col or not feature_cols:
        return jsonify({'error': 'Veuillez sélectionner la variable cible et au moins une variable explicative.'}), 400

    # Vérifier si les colonnes existent
    for col in [target_col] + feature_cols:
        if col not in df.columns:
            return jsonify({'error': f"La colonne '{col}' est introuvable dans le jeu de données."}), 400
    
    df = df.copy()

    # ==============================================================================
    # NOUVEAU : TRAITEMENT DES VALEURS ABERRANTES (OUTLIERS)
    # ==============================================================================
    # On applique le traitement sur la variable cible et les variables explicatives
    cols_to_process = [target_col] + feature_cols
    
    for col in cols_to_process:
        # S'assurer que la colonne est bien numérique
        if pd.api.types.is_numeric_dtype(df[col]):
            # 1. Calcul des bornes avec la méthode IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 2. Calcul de la médiane qui servira au remplacement
            median_value = df[col].median()
            
            # 3. Identification des index des outliers
            outlier_indexes = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            
            # 4. Remplacement des outliers par la médiane
            if not outlier_indexes.empty:
                df.loc[outlier_indexes, col] = median_value
    # ==============================================================================
    # FIN DU BLOC DE TRAITEMENT DES VALEURS ABERRANTES
    # ==============================================================================


    # Ingénierie des fonctionnalités
    df['heure'] = df[time_col].dt.hour
    df['jour_semaine'] = df[time_col].dt.dayofweek
    df['mois'] = df[time_col].dt.month
    df['jour'] = df[time_col].dt.day
    df['minute'] = df[time_col].dt.minute
    
    df.sort_values(by=time_col, inplace=True)
    
    # Ajout des variables de "lag"
    df[f'{target_col}_lag1'] = df[target_col].shift(1)
    
    for col in feature_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    
    # Remplacer les valeurs manquantes (NaN) par la médiane pour ne pas perdre de lignes
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)

    if df.empty:
        return jsonify({'error': 'Le jeu de données est trop petit pour la modélisation après la création des variables de lag.'}), 400

    # Définition des variables explicatives (features) et de la variable cible (target)
    features_with_time_and_lag = [f'{target_col}_lag1'] + [f'{col}_lag1' for col in feature_cols] + ['heure', 'jour_semaine', 'mois', 'jour', 'minute']
    features_final = [col for col in features_with_time_and_lag if col in df.columns]

    X = df[features_final]
    y = df[target_col]

    # Division des données en ensembles d'entraînement et de test
    if len(df) < 2:
        return jsonify({'error': 'Le jeu de données doit contenir au moins deux lignes de données valides pour la modélisation.'}), 400

    split_point = int(len(df) * 0.8)
    if split_point == 0:
        split_point = 1
        
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    if X_train.empty or X_test.empty:
        return jsonify({'error': 'Les ensembles de données d\'entraînement ou de test sont vides. Le jeu de données est trop petit.'}), 400

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        model_filename = f'modele_{name.replace(" ", "_")}.joblib'
        joblib.dump(model, os.path.join(app.config['OUTPUT_FOLDER'], model_filename))

        results[name] = {
            'mse': mse,
            'r2': r2,
            'mae': mae
        }

    # Fonction pour la prédiction itérative
    def predict_next_hours(model_name, num_steps=18):
        model_path = os.path.join(app.config['OUTPUT_FOLDER'], f'modele_{model_name.replace(" ", "_")}.joblib')
        if not os.path.exists(model_path):
            return []
        
        model = joblib.load(model_path)
        last_row = df.tail(1).copy()
        
        predictions = []
        current_data = last_row.iloc[0].to_dict()

        for _ in range(num_steps):
            # Mettre à jour les variables de lag avec les valeurs de l'itération précédente
            current_data[f'{target_col}_lag1'] = current_data[target_col]
            for col in feature_cols:
                current_data[f'{col}_lag1'] = current_data[col]
            
            # Faire avancer le temps de 10 minutes
            current_data[time_col] += timedelta(minutes=10)
            
            # Mettre à jour les features temporelles
            current_data['heure'] = current_data[time_col].hour
            current_data['minute'] = current_data[time_col].minute
            current_data['jour_semaine'] = current_data[time_col].dayofweek
            current_data['jour'] = current_data[time_col].day
            current_data['mois'] = current_data[time_col].month

            # Créer le DataFrame pour la prédiction
            predict_df = pd.DataFrame([current_data])[features_final]
            
            # Faire la prédiction
            prediction_value = model.predict(predict_df)[0]
            
            # Stocker la prédiction
            predictions.append({
                'time': current_data[time_col].strftime('%Y-%m-%d %H:%M'),
                'value': prediction_value
            })
            
            # Mettre à jour la variable cible pour la prochaine itération
            current_data[target_col] = prediction_value

        return predictions
        
    predictions_rf = predict_next_hours("Random Forest")
    predictions_gb = predict_next_hours("Gradient Boosting")

    return jsonify({
        'message': 'Modèles entraînés et prédictions générées avec succès.',
        'results': results,
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


# ==============================================================================
# PAGE DE COMPARAISON METEO
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. ROUTE POUR AFFICHER LA PAGE HTML (MÉTHODE GET)
# ------------------------------------------------------------------------------
# Le but de cette route est de simplement retourner le fichier comparison.html.
# C'est l'URL que vous devez visiter dans votre navigateur.
#
@app.route('/comparaison')
def comparaison_page():
    """Affiche la page de comparaison météo."""
    return render_template('comparaison_page.html')


# ------------------------------------------------------------------------------
# 2. ROUTE POUR TRAITER LES DONNÉES ET GÉNÉRER LES GRAPHIQUES (MÉTHODE POST)
# ------------------------------------------------------------------------------
# Cette route est appelée par le JavaScript lorsque l'utilisateur clique sur le bouton.
# Elle attend des données JSON et ne peut pas être accédée directement dans le navigateur.
#
@app.route('/compare_with_meteo', methods=['POST']) # <-- L'ajout de methods=['POST'] est CRUCIAL
def compare_with_meteo():
    """
    Compare les données du capteur avec les données de Meteostat.
    La ville est fournie directement par l'utilisateur.
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
        return jsonify({'error': "Veuillez sélectionner les colonnes et saisir un nom de ville."}), 400

    # Préparation des données utilisateur
    df_prepared = df.copy()
    df_prepared[time_col] = pd.to_datetime(df_prepared[time_col])
    
    start_date = df_prepared[time_col].min()
    end_date = df_prepared[time_col].max()

    # Géolocalisation de la ville
    try:
        geolocator = Nominatim(user_agent="pollugard-app-v3", timeout=10)
        location = geolocator.geocode(city_name)
        if location is None:
            return jsonify({'error': f"La ville '{city_name}' n'a pas pu être géolocalisée."}), 400
        meteo_point = Point(location.latitude, location.longitude)
    except Exception as e:
        return jsonify({'error': f"Erreur de géolocalisation : {e}"}), 500

    # Récupération des données Meteostat
    try:
        meteo_data = Hourly(meteo_point, start_date, end_date)
        meteo_df = meteo_data.fetch()
        if meteo_df.empty:
            return jsonify({'message': f"Aucune donnée météo trouvée pour {city_name} entre {start_date.date()} et {end_date.date()}."}), 200
    except Exception as e:
        return jsonify({'error': f"Erreur lors de la récupération des données Meteostat : {e}"}), 500

    # Fusion et alignement des données
    local_timezone = 'Europe/Paris' # À adapter si nécessaire
    # Ligne corrigée
    meteo_df.index = meteo_df.index.tz_localize('UTC').tz_convert(local_timezone)
    df_prepared[time_col] = df_prepared[time_col].dt.tz_localize(local_timezone, ambiguous='infer')
    df_indexed = df_prepared.set_index(time_col)
    
    # Utiliser numeric_only=True pour éviter les avertissements sur les colonnes non-numériques
    df_resampled = df_indexed.resample('H').mean(numeric_only=True) 

    comparison_df = pd.merge(df_resampled, meteo_df, left_index=True, right_index=True, how='inner')

    if comparison_df.empty:
        return jsonify({'message': 'Aucun chevauchement temporel trouvé entre vos données et les données météo.'}), 200

    # Création des graphiques Plotly
    plots = {}

    # Graphique Température
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[temp_col], name='Température Capteur', mode='lines'))
    fig_temp.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['temp'], name='Température Meteostat', mode='lines'))
    fig_temp.update_layout(title=f'Comparaison de Température ({city_name})', xaxis_title='Date', yaxis_title='Température (°C)')
    plots['temperature'] = fig_temp.to_json()

    # Graphique Humidité
    fig_humidity = go.Figure()
    fig_humidity.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[humidity_col], name='Humidité Capteur', mode='lines'))
    fig_humidity.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['rhum'], name='Humidité Meteostat', mode='lines'))
    fig_humidity.update_layout(title=f'Comparaison de l\'Humidité ({city_name})', xaxis_title='Date', yaxis_title='Humidité Relative (%)')
    plots['humidity'] = fig_humidity.to_json()

    return jsonify({
        'status': 'success',
        'city': city_name,
        'plots': plots
    })

if __name__ == '__main__':
    app.run(debug=True)
