import streamlit as st
import pandas as pd
import joblib
import os


# ========== 🔑 API Key Validation (Required for Render) ===========
API_KEY = os.environ.get("API_KEY", "")

if API_KEY != os.environ.get("RENDER_API_KEY"):
    st.error("🚫 Invalid API Key. Please set the correct API_KEY in your environment variables.")
    st.stop()


# ========== 🎯 Chargement du modèle et des features entraînés ==========
model = joblib.load("random_forest_model.pkl")
model_features = joblib.load("model_features.pkl")  # liste des colonnes vues à l'entraînement

# ========== 🎨 Configuration de la page ==========
st.set_page_config(page_title="🎓 Prédiction de Complétion de Cours", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #f3f6fa;
            color: #343a40;
        }
        .stButton>button {
            background-color: #2961ff;
            color: white;
            border-radius: 8px;
            padding: 10px 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== 🧠 Titre & Instructions ==========
st.title("🎯 Prédire la Complétion d’un Cours en Ligne")
st.markdown("Renseignez les informations d’un étudiant pour prédire s’il **terminera ou abandonnera** un cours en ligne.")

# ========== 📥 Entrées Utilisateur ==========
st.header("🧾 Informations sur l'étudiant")

TimeSpentOnCourse = st.slider("Temps passé sur le cours (heures)", 0.0, 100.0, 25.0)
NumberOfVideosWatched = st.slider("Nombre de vidéos regardées", 0, 20, 5)
NumberOfQuizzesTaken = st.slider("Nombre de quiz pris", 0, 10, 2)
QuizScores = st.slider("Score moyen aux quiz (%)", 0.0, 100.0, 75.0)
CompletionRate = st.slider("Taux de complétion actuel (%)", 0.0, 100.0, 60.0)

DeviceType = st.radio("Type d'appareil utilisé :", ["Desktop", "Mobile"])
device_code = 0 if DeviceType == "Desktop" else 1

course_category = st.selectbox("Catégorie du cours", ["Programming", "Science", "Health", "Business", "Arts"])

# ========== 🛠️ Construction du vecteur de features ==========
input_dict = {
    "TimeSpentOnCourse": [TimeSpentOnCourse],
    "NumberOfVideosWatched": [NumberOfVideosWatched],
    "NumberOfQuizzesTaken": [NumberOfQuizzesTaken],
    "QuizScores": [QuizScores],
    "CompletionRate": [CompletionRate],
    "DeviceType": [device_code],
    f"CourseCategory_{course_category}": [1]
}

# Ajouter les colonnes manquantes attendues par le modèle
for col in model_features:
    if col not in input_dict:
        input_dict[col] = [0]

# Organiser les colonnes dans le bon ordre
X_input = pd.DataFrame(input_dict)[model_features]

# ========== 🔍 Prédiction ==========
if st.button("🔮 Prédire la complétion"):
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    if pred == 1:
        st.success(f"✅ L'étudiant est **susceptible de terminer** le cours. (Probabilité : {proba:.2f})")
    else:
        st.error(f"⚠️ L'étudiant est **à risque d'abandon**. (Probabilité : {proba:.2f})")
