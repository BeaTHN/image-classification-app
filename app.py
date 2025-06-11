import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import sqlite3
import datetime
import io
import pandas as pd
from passlib.context import CryptContext
import requests
import time
import math # Pour la pagination

# --- Configuration de la page ---
st.set_page_config(layout="wide", page_title="Prédiction Image")

# --- Configuration ---
MODEL_PATH = "final_model_120races.keras"
BREED_LIST_PATH = "breed_list.json"
IMG_SIZE = (224, 224)
IMAGE_DIR = "images"
DB_PATH = "predictions.db"
DOG_API_BASE_URL = "https://dog.ceo/api"
ADMIN_USERNAME = "admin"
API_TIMEOUT = 5
API_RETRIES = 2
API_RETRY_DELAY = 1
HISTORY_PAGE_SIZE = 10 # Nombre d'éléments par page dans l'historique

# Contexte pour le hachage
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Créer les répertoires
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# --- Fonctions Base de Données (SQLite) ---
# ... (init_db, verify_password, get_user, create_user, get_all_users, delete_user, save_prediction, update_feedback - inchangées depuis v12) ...
def init_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, hashed_password TEXT NOT NULL)")
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'image_blob' not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN image_blob BLOB")
            print("Colonne 'image_blob' ajoutée à 'predictions'.")
        cursor.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, username TEXT, image_path TEXT, predicted_breed TEXT, confidence_score REAL, feedback TEXT CHECK(feedback IN ('correct', 'incorrect', 'unsure')), image_blob BLOB)")
        cursor.execute("SELECT username FROM users WHERE username = ?", (ADMIN_USERNAME,))
        if cursor.fetchone() is None:
            hashed_password = pwd_context.hash("password123")
            cursor.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (ADMIN_USERNAME, hashed_password))
            print(f"Utilisateur '{ADMIN_USERNAME}' par défaut créé.")
        conn.commit()
        conn.close()
        print(f"DB initialisée/vérifiée: {db_path}")
    except sqlite3.Error as e:
        print(f"Erreur SQLite (init_db): {e}")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db_path, username):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT username, hashed_password FROM users WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        conn.close()
        return user_data
    except sqlite3.Error as e:
        print(f"Erreur SQLite (get_user): {e}"); return None

def create_user(db_path, username, password):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        hashed_password = pwd_context.hash(password)
        cursor.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()
        print(f"Utilisateur '{username}' créé.")
        return True
    except sqlite3.IntegrityError:
        print(f"Erreur: Nom d'utilisateur '{username}' déjà pris."); conn.close(); return False
    except sqlite3.Error as e:
        print(f"Erreur SQLite (create_user): {e}"); conn.close(); return False

def get_all_users(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users ORDER BY username")
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        return users
    except sqlite3.Error as e:
        print(f"Erreur SQLite (get_all_users): {e}"); return []

def delete_user(db_path, username):
    if username == ADMIN_USERNAME:
        print("Erreur: Impossible de supprimer l'utilisateur admin."); return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        if rows_affected > 0:
            print(f"Utilisateur '{username}' supprimé."); return True
        else:
            print(f"Utilisateur '{username}' non trouvé pour suppression."); return False
    except sqlite3.Error as e:
        print(f"Erreur SQLite (delete_user): {e}"); conn.close(); return False

def save_prediction(db_path, username, image_bytes, predicted_breed, confidence_score):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (username, image_path, predicted_breed, confidence_score, image_blob) VALUES (?, ?, ?, ?, ?)", (username, None, predicted_breed, confidence_score, image_bytes))
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"Prédiction (ID: {prediction_id}) et image BLOB sauvegardées.")
        return prediction_id
    except sqlite3.Error as e:
        print(f"Erreur SQLite (save_prediction): {e}"); return None

def update_feedback(db_path, prediction_id, feedback):
    allowed_feedback = ['correct', 'incorrect', 'unsure']
    if feedback not in allowed_feedback: return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE predictions SET feedback = ? WHERE id = ?", (feedback, prediction_id))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        return rows_affected > 0
    except sqlite3.Error as e:
        print(f"Erreur SQLite (update_feedback): {e}"); return False

# Modifié pour la pagination
def get_user_predictions(db_path, username, page_number=1, page_size=10):
    """Récupère une page de l'historique avec pagination.
       Retourne: (DataFrame de la page, nombre total de prédictions)
    """
    offset = (page_number - 1) * page_size
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Compter le total des prédictions pour cet utilisateur
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE username = ?", (username,))
        total_predictions = cursor.fetchone()[0]

        # Récupérer la page actuelle
        query = """
            SELECT
                id, timestamp AS Date, predicted_breed AS Race_Prédite,
                printf('%.2f%%', confidence_score) AS Confiance,
                feedback AS Feedback, image_blob AS ImageBlob
            FROM predictions
            WHERE username = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, (username, page_size, offset))
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()

        if not rows:
            df = pd.DataFrame(columns=column_names)
        else:
            df = pd.DataFrame(rows, columns=column_names)

        print(f"Historique page {page_number} (taille {page_size}) récupéré pour {username}. Total: {total_predictions}")
        return df, total_predictions

    except sqlite3.Error as e:
        print(f"Erreur SQLite (get_user_predictions): {e}")
        return pd.DataFrame(), 0
    except Exception as e:
        print(f"Erreur inattendue (get_user_predictions): {e}")
        return pd.DataFrame(), 0

# --- Fonctions Utilitaires (Modèle, Races, Image, API Dog) ---
# ... (load_keras_model, load_breed_list, preprocess_image, get_reference_image_url - inchangées depuis v12) ...
@st.cache_resource
def load_keras_model(model_path):
    try:
        script_dir = os.path.dirname(__file__)
        abs_model_path = os.path.join(script_dir, model_path)
        if not os.path.exists(abs_model_path): print(f"Erreur critique : Modèle introuvable : {abs_model_path}"); return None
        model = tf.keras.models.load_model(abs_model_path)
        return model
    except Exception as e:
        print(f"Erreur critique (load_keras_model) : {e}"); return None

@st.cache_data
def load_breed_list(json_path):
    try:
        script_dir = os.path.dirname(__file__)
        abs_json_path = os.path.join(script_dir, json_path)
        if not os.path.exists(abs_json_path): print(f"Erreur critique : JSON introuvable : {abs_json_path}"); return []
        with open(abs_json_path, 'r') as f: breed_list = json.load(f)
        cleaned_breed_list = [breed.replace('_', ' ').replace('-', ' ').title() for breed in breed_list]
        return cleaned_breed_list
    except Exception as e:
        print(f"Erreur critique (load_breed_list) : {e}"); return []

def preprocess_image(image_pil, img_size):
    try:
        if image_pil.mode != 'RGB': image_pil = image_pil.convert('RGB')
        img = image_pil.resize(img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        st.error(f"Erreur (preprocess_image) : {e}"); return None

@st.cache_data(ttl=3600)
def get_reference_image_url(breed_name):
    parts = breed_name.lower().split()
    if len(parts) > 1:
        api_breed = f"{parts[-1]}/{''.join(parts[:-1])}"
        api_breed_alt = f"{parts[-1]}/{'-'.join(parts[:-1])}"
        api_breed_simple = parts[-1]
    else:
        api_breed = parts[0]; api_breed_alt = None; api_breed_simple = None
    formats_to_try = [api_breed]
    if api_breed_alt and api_breed_alt != api_breed: formats_to_try.append(api_breed_alt)
    if api_breed_simple and api_breed_simple != api_breed: formats_to_try.append(api_breed_simple)
    last_error = None
    for attempt in range(API_RETRIES + 1):
        for fmt in formats_to_try:
            api_url = f"{DOG_API_BASE_URL}/breed/{fmt}/images/random"
            print(f"Tentative {attempt+1}/{API_RETRIES+1} - Appel API Dog: {api_url}")
            try:
                response = requests.get(api_url, timeout=API_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success": return data.get("message"), None
                else: last_error = f"Race '{breed_name}' (format: {fmt}) non trouvée par l'API Dog."; print(last_error)
            except requests.exceptions.Timeout: last_error = f"Timeout API Dog ({api_url})."; print(last_error); break
            except requests.exceptions.HTTPError as http_err: last_error = f"Erreur HTTP {http_err.response.status_code} API Dog ({api_url})."; print(last_error)
            except requests.exceptions.RequestException as req_err: last_error = f"Erreur connexion API Dog ({api_url}): {req_err}."; print(last_error); break
            except Exception as e: last_error = f"Erreur inattendue API Dog ({api_url}): {e}."; print(last_error); break
        if last_error and ("connexion" in last_error or "Timeout" in last_error or "inattendue" in last_error):
            if attempt < API_RETRIES: print(f"Attente {API_RETRY_DELAY}s..."); time.sleep(API_RETRY_DELAY)
            else: return None, f"Impossible de contacter l'API Dog après {API_RETRIES+1} tentatives."
        elif last_error: break
    if last_error and "non trouvée" in last_error: return None, f"Impossible de trouver image de référence pour '{breed_name}'."
    elif last_error: return None, last_error
    else: return None, f"Erreur inconnue recherche image pour '{breed_name}'."

# --- Système d'Authentification (Connexion uniquement) ---
def authentication_form():
    # ... (inchangée depuis v12) ...
    if "logged_in" not in st.session_state: st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.title("🔒 Connexion")
        username = st.text_input("Nom d'utilisateur", key="login_username")
        password = st.text_input("Mot de passe", type="password", key="login_password")
        if st.button("Se connecter", key="login_button"):
            user_data = get_user(DB_PATH, username)
            if user_data:
                stored_username, stored_hashed_password = user_data
                if verify_password(password, stored_hashed_password):
                    st.session_state.logged_in = True; st.session_state.username = stored_username; st.rerun()
                else: st.error("Nom d'utilisateur ou mot de passe incorrect.")
            else: st.error("Nom d'utilisateur ou mot de passe incorrect.")
        st.info("Si vous n'avez pas de compte, veuillez contacter l'administrateur.")
        return False
    return True

# --- Initialisation ---
init_db(DB_PATH)
model = load_keras_model(MODEL_PATH)
breed_list = load_breed_list(BREED_LIST_PATH)

# Initialiser l'état de la page pour l'historique si nécessaire
if 'history_page' not in st.session_state:
    st.session_state.history_page = 1

# --- Interface Streamlit ---
if authentication_form():
    current_username = st.session_state.get('username', 'N/A')
    is_admin = (current_username == ADMIN_USERNAME)

    st.sidebar.write(f"Connecté: **{current_username}**")
    if st.sidebar.button("Se déconnecter", key="logout_button"):
        # Réinitialiser la page d'historique lors de la déconnexion
        st.session_state.history_page = 1
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.session_state.logged_in = False; st.rerun()

    st.sidebar.title("Menu Principal")
    menu_options = ["Accueil", "Prédiction de race", "Historique", "Santé"]
    if is_admin: menu_options.append("Gestion Utilisateurs")
    app_mode = st.sidebar.radio("Section", menu_options, disabled=(model is None or not breed_list))

    # Réinitialiser la page d'historique si on quitte la section Historique
    if app_mode != "Historique" and st.session_state.history_page != 1:
         st.session_state.history_page = 1

    # --- Contenu Principal ---
    if app_mode == "Accueil":
        # ... (inchangé) ...
        st.title("Bienvenue - Prédiction Image")
        st.markdown("Connectez-vous pour utiliser l'application.")
        st.header("Fonctionnalités")
        st.markdown("- **Prédiction de Race**: Identifiez la race d'un chien.\n- **Feedback**: Donnez votre avis.\n- **Historique**: Consultez vos prédictions.\n- **Santé**: Outils d'analyse (en développement).")
        if is_admin: st.markdown("- **Gestion Utilisateurs**: Ajoutez ou supprimez des utilisateurs.")

    elif app_mode == "Prédiction de race":
        # ... (inchangé depuis v12) ...
        if model is None or not breed_list: st.error("Erreur chargement modèle/liste.")
        else:
            st.title("🐾 Prédiction de Race 🐶")
            uploaded_file = st.file_uploader("🦴 Choisir image...", type=["jpg", "jpeg", "png"], key="file_uploader")
            if uploaded_file is not None and uploaded_file != st.session_state.get('last_uploaded_file'):
                st.session_state.feedback_given = False; st.session_state.last_prediction_id = None; st.session_state.last_predicted_breed = None; st.session_state.last_uploaded_file = uploaded_file
            elif uploaded_file is None: st.session_state.last_uploaded_file = None; st.session_state.last_predicted_breed = None
            if uploaded_file is not None:
                col1, col2 = st.columns([1, 2])
                with col1:
                    try:
                        image_bytes = uploaded_file.getvalue(); image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption='Image téléchargée', use_column_width=True)
                    except Exception as e: st.error(f"Erreur affichage image: {e}"); st.stop()
                with col2:
                    if 'last_prediction_id' not in st.session_state or st.session_state.last_prediction_id is None:
                        with st.spinner('Prédiction...'):
                            processed_image = preprocess_image(image, IMG_SIZE)
                            if processed_image is not None:
                                try:
                                    prediction = model.predict(processed_image); score = prediction[0]; predicted_index = np.argmax(score); confidence = 100 * np.max(score)
                                    if 0 <= predicted_index < len(breed_list):
                                        predicted_breed = breed_list[predicted_index]; st.session_state.last_predicted_breed = predicted_breed
                                        st.subheader("Résultat :"); st.success(f"Race: **{predicted_breed}**"); st.info(f"Confiance: **{confidence:.2f}%**")
                                        prediction_id = save_prediction(DB_PATH, st.session_state.username, image_bytes, predicted_breed, confidence)
                                        if prediction_id: st.session_state.last_prediction_id = prediction_id; st.session_state.feedback_given = False
                                        else: st.warning("Sauvegarde historique échouée.")
                                except Exception as e: st.error(f"Erreur prédiction/sauvegarde: {e}")
                            else: st.error("Erreur de prétraitement de l'image.")
                    if 'last_predicted_breed' in st.session_state and st.session_state.last_predicted_breed:
                        predicted_breed_for_ref = st.session_state.last_predicted_breed
                        st.subheader(f"Exemple de {predicted_breed_for_ref} :"); ref_image_url, error_msg = get_reference_image_url(predicted_breed_for_ref)
                        if error_msg: st.warning(f"API Dog: {error_msg}")
                        elif ref_image_url: st.image(ref_image_url, caption=f"Image de référence pour {predicted_breed_for_ref}", use_column_width=True)
                    if 'last_prediction_id' in st.session_state and st.session_state.last_prediction_id is not None:
                        prediction_id = st.session_state.last_prediction_id; feedback_given = st.session_state.get('feedback_given', False)
                        if not feedback_given:
                            st.write("Prédiction correcte ?"); cols_fb = st.columns(3)
                            with cols_fb[0]:
                                if st.button("👍 Oui", key=f"fb_correct_{prediction_id}"):
                                    if update_feedback(DB_PATH, prediction_id, 'correct'): st.session_state.feedback_given = True; st.success("Merci!"); st.rerun()
                                    else: st.error("Erreur feedback.")
                            with cols_fb[1]:
                                if st.button("👎 Non", key=f"fb_incorrect_{prediction_id}"):
                                    if update_feedback(DB_PATH, prediction_id, 'incorrect'): st.session_state.feedback_given = True; st.success("Merci!"); st.rerun()
                                    else: st.error("Erreur feedback.")
                            with cols_fb[2]:
                                if st.button("❓ Incertain", key=f"fb_unsure_{prediction_id}"):
                                    if update_feedback(DB_PATH, prediction_id, 'unsure'): st.session_state.feedback_given = True; st.success("Merci!"); st.rerun()
                                    else: st.error("Erreur feedback.")
                        else: st.info("Feedback enregistré.")

    elif app_mode == "Historique":
        st.title("📜 Historique des Prédictions")
        username = st.session_state.get('username')
        if username:
            # Récupérer la page actuelle et le total
            current_page = st.session_state.history_page
            history_df_page, total_predictions = get_user_predictions(DB_PATH, username, page_number=current_page, page_size=HISTORY_PAGE_SIZE)

            if total_predictions == 0:
                st.info("Aucun historique de prédiction pour le moment.")
            else:
                total_pages = math.ceil(total_predictions / HISTORY_PAGE_SIZE)
                st.write(f"Affichage de la page {current_page} sur {total_pages} ({total_predictions} prédictions au total)")

                # Afficher le DataFrame de la page
                st.dataframe(history_df_page[['Date', 'Race_Prédite', 'Confiance', 'Feedback']], use_container_width=True)

                # Afficher les détails pour la page actuelle
                st.subheader("Détails de la page")
                if not history_df_page.empty:
                    for index, row in history_df_page.iterrows():
                        with st.expander(f"{row['Date']} - {row['Race_Prédite']}"):
                            st.write(f"**Race:** {row['Race_Prédite']}")
                            st.write(f"**Confiance:** {row['Confiance']}")
                            st.write(f"**Feedback:** {row['Feedback'] if row['Feedback'] else 'N/A'}")
                            image_blob = row['ImageBlob']
                            if image_blob:
                                try: st.image(image_blob, width=200)
                                except Exception as img_err: st.warning(f"Impossible d'afficher l'image (ID: {row['id']}): {img_err}")
                            else: st.warning(f"Données image manquantes (ID: {row['id']})")
                else:
                    st.info("Aucune prédiction sur cette page.") # Ne devrait pas arriver si total_predictions > 0

                # Contrôles de pagination
                st.write("Navigation:")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("⬅️ Précédent", disabled=(current_page <= 1)):
                        st.session_state.history_page -= 1
                        st.rerun()
                with col2:
                    # Afficher le numéro de page (peut être remplacé par un selectbox si beaucoup de pages)
                    st.write(f"Page {current_page}/{total_pages}")
                with col3:
                    if st.button("Suivant ➡️", disabled=(current_page >= total_pages)):
                        st.session_state.history_page += 1
                        st.rerun()
        else:
            st.error("Utilisateur non identifié.")

    elif app_mode == "Santé":
        # ... (inchangé depuis v12) ...
        st.title("🩺 Section Santé Animale"); st.warning("🚧 En développement... 🚧")
        st.markdown("Objectif: outils d'analyse IA pour la santé animale.")
        st.subheader("Fonctionnalités prévues:"); st.markdown("- Analyse lésions cutanées\n- Analyse symptômes\n- Suivi poids/activité")
        st.info("Plus à venir. Suggestions bienvenues !")

    elif app_mode == "Gestion Utilisateurs" and is_admin:
        # ... (inchangé depuis v12) ...
        st.title("🔑 Gestion des Utilisateurs")
        st.subheader("Ajouter un nouvel utilisateur")
        with st.form("add_user_form", clear_on_submit=True):
            new_username = st.text_input("Nom d'utilisateur")
            new_password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Ajouter Utilisateur")
            if submitted:
                if not new_username or not new_password: st.warning("Veuillez fournir un nom d'utilisateur et un mot de passe.")
                elif len(new_password) < 6: st.warning("Le mot de passe doit faire au moins 6 caractères.")
                else:
                    if get_user(DB_PATH, new_username) is None:
                        if create_user(DB_PATH, new_username, new_password): st.success(f"Utilisateur '{new_username}' ajouté avec succès !")
                        else: st.error("Erreur lors de la création de l'utilisateur.")
                    else: st.error(f"Le nom d'utilisateur '{new_username}' existe déjà.")
        st.subheader("Liste des utilisateurs existants")
        all_users = get_all_users(DB_PATH)
        if all_users:
            cols = st.columns([3, 1]); cols[0].write("**Nom d'utilisateur**"); cols[1].write("**Action**")
            for user in all_users:
                cols = st.columns([3, 1]); cols[0].write(user)
                if user != ADMIN_USERNAME:
                    if cols[1].button("Supprimer", key=f"delete_{user}"):
                        if delete_user(DB_PATH, user): st.success(f"Utilisateur '{user}' supprimé."); st.rerun()
                        else: st.error(f"Erreur lors de la suppression de '{user}'.")
                else: cols[1].write("(Admin)")
        else: st.info("Aucun utilisateur trouvé (à part l'admin initial).")

# Gérer échec chargement initial
elif model is None or not breed_list:
    st.error("Erreur critique au démarrage (modèle/liste).")


