import streamlit as st
import pandas as pd
import joblib

# Chargement des modèles
model_grain_mais = joblib.load("model_grain_mais.pkl")


#Chargement des fonctions utiles
def make_prediction(nouvelle_observation):
    # Faire la prédiction avec le modèle
    prediction = model_grain_mais.predict(nouvelle_observation)
    return prediction[0]


st.title("IAGRIKOL🌱") #titre de l'application
st.sidebar.title("Choisir la culture à prédire") #Création des onglets
choix = st.sidebar.radio("Sélectionne la culture :", ["Modèle 1", "Modèle 2", "Modèle 3"])

if choix=="Modèle 1":
    st.header("Modèle 1")
    st.write("Ici, tu peux insérer le modèle 1")

if choix=="Modèle 2":
    # Création de deux colonnes
    col1, col2 = st.columns(2)

# 📌 **Colonne 1 : Saisie des valeurs numériques**
    with col1:
        st.subheader("Paramètres physiques du sol à 30 cm de profondeur")
        clay = st.number_input("Teneur en argile", min_value=0.0, max_value=100.0, value=0.0)
        silt = st.number_input("Teneur de limon", min_value=0.0, max_value=100.0, value=0.0)
        sand = st.number_input("Teneur en sable", min_value=0.0, max_value=100.0, value=0.0)
        ph = st.number_input("pH du sol", min_value=0.0, max_value=14.0, value=7.0)
        organic_carbon = st.number_input("Teneur en carbone organique:", min_value=0.0, max_value=10.0, value=0.0)
        bdod = st.number_input("Densité apparente", min_value=0.0, max_value=5.0, value=0.0)
        cfvo = st.number_input("Matière organique", min_value=0.0, max_value=100.0, value=0.0)
    with col2:
        st.subheader("Présence d'autres cultures sur le sol")
        ble_tendre = st.radio("Blé tendre:", ["Oui", "Non"], index=1,horizontal=True)
        ble_dur = st.radio("Blé dur:", ["Oui", "Non"], index=1,horizontal=True)
        orge = st.radio("Orge:", ["Oui", "Non"], index=1,horizontal=True)
        colza = st.radio("Colza:", ["Oui", "Non"], index=1,horizontal=True)
        ensilage_mais = st.radio("Ensilage mais:", ["Oui", "Non"], index=1,horizontal=True)
        tournsol = st.radio("Tournesol:", ["Oui", "Non"], index=1,horizontal=True)
        bettrave_a_sucre = st.radio("Bettrave à sucre:", ["Oui", "Non"], index=1,horizontal=True)
        vignobles=st.radio("Vignoble:", ["Oui", "Non"], index=1,horizontal=True)

    if st.button("Prédire"):
        st.success(f"Valeurs enregistrées ✅\n\nClay: {clay}, pH: {ph}, Blé tendre: {ble_tendre}")
        nouvelle_observation_dict = {
            "clay_0to30cm_percent": clay,  # Argile (0-30 cm)
            "silt_0to30cm_percent": silt,  # Limon (0-30 cm)
            "sand_0to30cm_percent": sand,  # Sable (0-30 cm)
            "ph_h2o_0to30cm": ph,  # pH (H2O, 0-30 cm)
            "organic_carbon_0to30cm_percent": organic_carbon,  # Carbone organique (0-30 cm)
            "bdod_0to30cm": bdod,  # Densité apparente (0-30 cm)
            "cfvo_0to30cm_percent": cfvo,
            "ble_tendre": 1 if ble_tendre == "Oui" else 0,  # Blé tendre
            "ble_dur": 1 if ble_dur == "Oui" else 0,  # Blé dur
            "ensilage_mais": 1 if ensilage_mais == "Oui" else 0,  # Ensilage de maïs
            "orge": 1 if orge == "Oui" else 0,  # Orge
            "colza": 1 if colza == "Oui" else 0,  # Colza
            "tournsol": 1 if tournsol == "Oui" else 0,  # Tournesol
            "bettrave_a_sucre": 1 if bettrave_a_sucre == "Oui" else 0,  # Betterave à sucre
            "vignobles": 1 if vignobles == "Oui" else 0  # Vignobles
        }
        # Convertir en DataFrame
        nouvelle_observation = pd.DataFrame([nouvelle_observation_dict])
    
        # Prédiction et afficher le résultat
        prediction = make_prediction(nouvelle_observation)
        st.write(f"Prédiction du rendement du maïs : {prediction}")