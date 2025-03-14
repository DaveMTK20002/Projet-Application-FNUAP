import streamlit as st
import pandas as pd
import joblib

# Chargement des modèles
model_grain_mais = joblib.load("model_grain_mais.pkl")
model_grain_mais_num = joblib.load("model_grain_mais_num.pkl")
model_ble_tendre = joblib.load("model_ble_tendre.pkl")
model_ble_tendre_num = joblib.load("model_ble_tendre_num.pkl")
#model_orge = joblib.load("model_orge.pkl")
#model_orge_num = joblib.load("model_orge_num.pkl")
#model_tournsol = joblib.load("model_tournsol.pkl")
#model_tournsol_num = joblib.load("model_tournsol_num.pkl")
#model_bettrave = joblib.load("model_bettrave.pkl")
#model_bettrave_num = joblib.load("model_bettrave_num.pkl")


#Chargement des fonctions utiles
def make_prediction_grain_mais(nouvelle_observation):
    prediction = model_grain_mais.predict_proba(nouvelle_observation)
    probabilite_adapte = prediction[:, 1]  # Probabilité que le sol soit adapté

    seuil = 0.85

    # Si la probabilité de la classe 1 est supérieure au seuil, prédire 1, sinon prédire 0
    prediction_ajustee = (probabilite_adapte > seuil).astype(int)
    return (prediction_ajustee[0],probabilite_adapte)



def make_prediction_grain_mais_num(nouvelle_observation):
    prediction = model_grain_mais_num.predict(nouvelle_observation)
    return prediction[0]


def make_prediction_ble_tendre(nouvelle_observation):
    prediction = model_ble_tendre.predict_proba(nouvelle_observation)
    probabilite_adapte = prediction[:, 1]  # Probabilité que le sol soit adapté

    seuil = 0.85

    # Si la probabilité de la classe 1 est supérieure au seuil, prédire 1, sinon prédire 0
    prediction_ajustee = (probabilite_adapte > seuil).astype(int)
    return (prediction_ajustee[0],probabilite_adapte)


def make_prediction_ble_tendre_num(nouvelle_observation):
    prediction = model_ble_tendre_num.predict(nouvelle_observation)
    return prediction[0]


def make_prediction_orge(nouvelle_observation):
    prediction = model_orge.predict_proba(nouvelle_observation)
    probabilite_adapte = prediction[:, 1]  # Probabilité que le sol soit adapté

    seuil = 0.85

    # Si la probabilité de la classe 1 est supérieure au seuil, prédire 1, sinon prédire 0
    prediction_ajustee = (probabilite_adapte > seuil).astype(int)
    return (prediction_ajustee[0],probabilite_adapte)



def make_prediction_orge_num(nouvelle_observation):
    prediction = model_orge_num.predict(nouvelle_observation)
    return prediction[0]


def make_prediction_tournsol(nouvelle_observation):
    prediction = model_tournsol.predict_proba(nouvelle_observation)
    probabilite_adapte = prediction[:, 1]  # Probabilité que le sol soit adapté

    seuil = 0.85

    # Si la probabilité de la classe 1 est supérieure au seuil, prédire 1, sinon prédire 0
    prediction_ajustee = (probabilite_adapte > seuil).astype(int)
    return (prediction_ajustee[0],probabilite_adapte)


def make_prediction_tournsol_num(nouvelle_observation):
    prediction = model_tournsol_num.predict(nouvelle_observation)
    return prediction[0]



def make_prediction_bettrave(nouvelle_observation):
    prediction = model_bettrave.predict_proba(nouvelle_observation)
    probabilite_adapte = prediction[:, 1]  # Probabilité que le sol soit adapté

    seuil = 0.85

    # Si la probabilité de la classe 1 est supérieure au seuil, prédire 1, sinon prédire 0
    prediction_ajustee = (probabilite_adapte > seuil).astype(int)
    return (prediction_ajustee[0],probabilite_adapte)



def make_prediction_bettrave_num(nouvelle_observation):
    prediction = model_bettrave_num.predict(nouvelle_observation)
    return prediction[0]



st.title("IAGRIKOL🌱") #titre de l'application
st.sidebar.title("Choisir la culture à prédire") #Création des onglets
choix = st.sidebar.radio("Sélectionne la culture :", ["Maïs", "Blé tendre", "Orge","Tournesol","Bettrave à sucre"])

if choix=="Blé tendre":
    st.header("Blé tendre")
    st.write("Entrez les caractéristiques du sol. Ensuite, validez avec le bouton Prédire.\nL'algorithme va estimer si ce sol est adpaté à la culture du Blé tendre ou pas.\nSi c'est le cas, l'algorithme pourra également estimer les rendements esomptés")
    



if choix=="Orge":
    st.header("Orge")
    st.write("Entrez les caractéristiques du sol. Ensuite, validez avec le bouton Prédire.\nL'algorithme va estimer si ce sol est adpaté à la culture de l'Orge ou pas.\nSi c'est le cas, l'algorithme pourra également estimer les rendements esomptés")
    


if choix=="Tournesol":
    st.header("Tournesol")
    st.write("Entrez les caractéristiques du sol. Ensuite, validez avec le bouton Prédire.\nL'algorithme va estimer si ce sol est adpaté à la culture du Tournesol ou pas.\nSi c'est le cas, l'algorithme pourra également estimer les rendements esomptés")
    


if choix=="Bettrave à sucre":
    st.header("Bettrave à sucre")
    st.write("Entrez les caractéristiques du sol. Ensuite, validez avec le bouton Prédire.\nL'algorithme va estimer si ce sol est adpaté à la culture de la Bettrave à sucre ou pas.\nSi c'est le cas, l'algorithme pourra également estimer les rendements esomptés")
    



if choix=="Maïs":
    st.header("Maïs")
    st.write("Entrez les caractéristiques du sol. Ensuite, validez avec le bouton Prédire.\nL'algorithme va estimer si ce sol est adpaté à la culture du mais ou pas.\nSi c'est le cas, l'algorithme pourra également estimer les rendements esomptés")
    # Création de deux colonnes
    col1, col2 = st.columns(2)

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
        orge = st.radio("Orge:", ["Oui", "Non"], index=1,horizontal=True)
        tournsol = st.radio("Tournesol:", ["Oui", "Non"], index=1,horizontal=True)
        bettrave_a_sucre = st.radio("Bettrave à sucre:", ["Oui", "Non"], index=1,horizontal=True)
    if st.button("Prédire"):
        st.success(f"Valeurs enregistrées ✅\n\nClay: {clay}, pH: {ph}, Blé tendre: {ble_tendre}")
        nouvelle_observation_dict = {
            "clay_0to30cm_percent": clay, 
            "silt_0to30cm_percent": silt, 
            "sand_0to30cm_percent": sand, 
            "ph_h2o_0to30cm": ph, 
            "organic_carbon_0to30cm_percent": organic_carbon, 
            "bdod_0to30cm": bdod, 
            "cfvo_0to30cm_percent": cfvo,
            "ble_tendre": 1 if ble_tendre == "Oui" else 0, 
            "orge": 1 if orge == "Oui" else 0, 
            "tournsol": 1 if tournsol == "Oui" else 0, 
            "bettrave_a_sucre": 1 if bettrave_a_sucre == "Oui" else 0
        }
        # Convertir en DataFrame
        nouvelle_observation = pd.DataFrame([nouvelle_observation_dict])
    
        # Prédiction et affichage du résultat
        prediction = make_prediction_grain_mais(nouvelle_observation)
        if prediction[0]==0:
            st.write(f"Sol non adapté pour le maïs. Probabilité de non adaptation: {1-prediction[1]}")
        else: 
            st.write(f"Sol adapté pour le maïs. Probabilité : {prediction[1]}")

            st.subheader("Superficie plantée pour autres cultures (en km2)")
            soft_wheat_area_km2=st.number_input("Ble tendre", min_value=0.0, value=0.0)
            barley_area_km2=st.number_input("Orge", min_value=0.0, value=0.0)
            sunflower_area_km2=st.number_input("Tournsol", min_value=0.0, value=0.0)
            sugarbeet_area_km2=st.number_input("Bettrave à sucre", min_value=0.0, value=0.0)
            
            nouvelle_observation_dict_num = {
            "clay_0to30cm_percent": clay, 
            "silt_0to30cm_percent": silt, 
            "sand_0to30cm_percent": sand, 
            "ph_h2o_0to30cm": ph, 
            "organic_carbon_0to30cm_percent": organic_carbon, 
            "bdod_0to30cm": bdod, 
            "cfvo_0to30cm_percent": cfvo,
            "soft_wheat_area_km2":soft_wheat_area_km2,
            "barley_area_km2":barley_area_km2,
            "sunflower_area_km2":sunflower_area_km2,
            "sugarbeet_area_km2":sugarbeet_area_km2
            }
            # Convertir en DataFrame
            nouvelle_observation_num = pd.DataFrame([nouvelle_observation_dict_num])
            prediction_num = make_prediction_grain_mais_num(nouvelle_observation_num)
            st.write(f"Estimation du rendement: {prediction_num[0]}")