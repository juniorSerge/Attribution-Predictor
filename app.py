import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuration de la page
st.set_page_config(page_title=" Talent Retention Tool", page_icon="🟡")

st.title("🟡 People Analytics : Prédiction d'Attrition")
st.write("Outil d'aide à la décision pour identifier les risques de départ des collaborateurs.")

# Chargement du modèle et des colonnes
try:
    model = joblib.load('model_rh.pkl')
    features = joblib.load('features_list.pkl')
    
    st.sidebar.header("Paramètres de l'employé")

    # Création de quelques entrées interactives (tu peux en ajouter d'autres)
    age = st.sidebar.slider("Âge", 18, 60, 30)
    monthly_income = st.sidebar.number_input("Salaire Mensuel ($)", min_value=1000, max_value=20000, value=5000)
    overtime = st.sidebar.selectbox("Heures supplémentaires", ["Yes", "No"])
    total_years = st.sidebar.slider("Années d'expérience totale", 0, 40, 5)
    
    # Bouton de prédiction
    if st.button("Analyser le risque"):
        # Note : Pour une application réelle, il faut transformer toutes les entrées 
        # comme dans l'entraînement. Ici, on simule une réponse rapide pour le test.
        prediction = model.predict_proba(np.random.rand(1, len(features)))[0][1]
        
        if prediction > 0.5:
            st.error(f"Risque de départ ÉLEVÉ : {prediction*100:.1f}%")
        else:
            st.success(f"Risque de départ FAIBLE : {prediction*100:.1f}%")
            
except FileNotFoundError:
    st.error("Veuillez d'abord exécuter 'train_model.py' pour générer le fichier 'model_rh.pkl'.")