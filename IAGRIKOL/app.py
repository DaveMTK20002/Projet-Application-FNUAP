import streamlit as st
import pandas as pd
import joblib
import os

model_path_grain = os.path.join(os.path.dirname(__file__), "model_grain_mais.pkl")
model_path_grain_num = os.path.join(os.path.dirname(__file__), "model_grain_mais_num.pkl")
model_path_ble_tendre = os.path.join(os.path.dirname(__file__), "model_ble_tendre.pkl")
model_path_ble_tendre_num = os.path.join(os.path.dirname(__file__), "model_ble_tendre_num.pkl")
model_path_orge = os.path.join(os.path.dirname(__file__), "model_orge.pkl")
model_path_orge_num = os.path.join(os.path.dirname(__file__), "model_orge_num.pkl")
model_path_bettrave = os.path.join(os.path.dirname(__file__), "model_bettrave_a_sucre.pkl")
model_path_bettrave_num = os.path.join(os.path.dirname(__file__), "model_bettrave_a_sucre_num.pkl")


# Chargement des modèles
model_grain_mais = joblib.load(model_path_grain)
model_grain_mais_num = joblib.load(model_path_grain_num)
model_ble_tendre = joblib.load(model_path_ble_tendre)
model_ble_tendre_num = joblib.load(model_path_ble_tendre_num)
model_orge = joblib.load(model_path_orge)
model_orge_num = joblib.load(model_path_orge_num)
model_bettrave = joblib.load(model_path_bettrave)
model_bettrave_num = joblib.load(model_path_bettrave_num)


#Chargement des fonctions utiles
def make_prediction_grain_mais(nouvelle_observation):
    prediction = model_grain_mais.predict_proba(nouvelle_observation)
    probabilite_adapte = prediction[:, 1]  # Probabilité que le sol soit adapté

    seuil = 0.70

    # Si la probabilité de la classe 1 est supérieure au seuil, prédire 1, sinon prédire 0
    prediction_ajustee = (probabilite_adapte > seuil).astype(int)
    return (prediction_ajustee[0],probabilite_adapte)



def make_prediction_grain_mais_num(nouvelle_observation):
    prediction = model_grain_mais_num.predict(nouvelle_observation)
    return prediction[0]


def make_prediction_ble_tendre(nouvelle_observation):
    prediction = model_ble_tendre.predict_proba(nouvelle_observation)
    probabilite_adapte = prediction[:, 1]  # Probabilité que le sol soit adapté

    seuil = 0.70

    # Si la probabilité de la classe 1 est supérieure au seuil, prédire 1, sinon prédire 0
    prediction_ajustee = (probabilite_adapte > seuil).astype(int)
    return (prediction_ajustee[0],probabilite_adapte)


def make_prediction_ble_tendre_num(nouvelle_observation):
    prediction = model_ble_tendre_num.predict(nouvelle_observation)
    return prediction[0]


def make_prediction_orge(nouvelle_observation):
    prediction = model_orge.predict_proba(nouvelle_observation)
    probabilite_adapte = prediction[:, 1]  # Probabilité que le sol soit adapté

    seuil = 0.70

    # Si la probabilité de la classe 1 est supérieure au seuil, prédire 1, sinon prédire 0
    prediction_ajustee = (probabilite_adapte > seuil).astype(int)
    return (prediction_ajustee[0],probabilite_adapte)



def make_prediction_orge_num(nouvelle_observation):
    prediction = model_orge_num.predict(nouvelle_observation)
    return prediction[0]





def make_prediction_bettrave(nouvelle_observation):
    prediction = model_bettrave.predict_proba(nouvelle_observation)
    probabilite_adapte = prediction[:, 1]  # Probabilité que le sol soit adapté

    seuil = 0.50

    # Si la probabilité de la classe 1 est supérieure au seuil, prédire 1, sinon prédire 0
    prediction_ajustee = (probabilite_adapte > seuil).astype(int)
    return (prediction_ajustee[0],probabilite_adapte)



def make_prediction_bettrave_num(nouvelle_observation):
    prediction = model_bettrave_num.predict(nouvelle_observation)
    return prediction[0]



st.title("IAGRIKOL🌱") #titre de l'application
st.sidebar.title("Choisir la culture à prédire") #Création des onglets
choix = st.sidebar.radio("Sélectionner la culture :", ["Maïs", "Blé tendre", "Orge","Bettrave à sucre"])

if choix=="Blé tendre":
    st.header("Blé tendre")
    st.write("Entrez les caractéristiques du sol. Ensuite, validez avec le bouton Prédire.\nL'algorithme va estimer si ce sol est adpaté à la culture du Blé tendre ou pas.\nSi c'est le cas, l'algorithme pourra également estimer les rendements esomptés")

    # Entrées utilisateur
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Paramètres physiques du sol à 30 cm de profondeur")
        clay = st.number_input("Teneur en argile", min_value=0.0, max_value=100.0, value=0.0)
        silt = st.number_input("Teneur de limon", min_value=0.0, max_value=100.0, value=0.0)
        sand = st.number_input("Teneur en sable", min_value=0.0, max_value=100.0, value=0.0)
        ph = st.number_input("pH du sol", min_value=0.0, max_value=14.0, value=7.0)
        organic_carbon = st.number_input("Teneur en carbone organique:", min_value=0.0, max_value=100.0, value=0.0)
        bdod = st.number_input("Densité apparente", min_value=0.0, max_value=5.0, value=0.0)
        cfvo = st.number_input("Matière organique", min_value=0.0, max_value=100.0, value=0.0)
    with col2:
        st.subheader("Présence d'autres cultures sur le sol")
        grain_mais = st.radio("Grain de mais:", ["Oui", "Non"], index=1,horizontal=True)
        orge = st.radio("Orge:", ["Oui", "Non"], index=1,horizontal=True)
        tournsol = st.radio("Tournesol:", ["Oui", "Non"], index=1,horizontal=True)
        bettrave_a_sucre = st.radio("Bettrave à sucre:", ["Oui", "Non"], index=1,horizontal=True)
    nouvelle_observation_dict = {
            "clay_0to30cm_percent": clay, 
            "silt_0to30cm_percent": silt, 
            "sand_0to30cm_percent": sand, 
            "ph_h2o_0to30cm": ph, 
            "organic_carbon_0to30cm_percent": organic_carbon, 
            "bdod_0to30cm": bdod, 
            "cfvo_0to30cm_percent": cfvo,
            "grain_mais": 1 if grain_mais == "Oui" else 0, 
            "orge": 1 if orge == "Oui" else 0, 
            "tournsol": 1 if tournsol == "Oui" else 0, 
            "bettrave_a_sucre": 1 if bettrave_a_sucre == "Oui" else 0
        }
    nouvelle_observation = pd.DataFrame([nouvelle_observation_dict])

    # Bouton de prédiction
    if st.button("Prédire"):
        st.session_state['prediction_done'] = True
        st.session_state['estimation_done'] = False
        prediction = make_prediction_ble_tendre(nouvelle_observation)
        st.session_state['prediction'] = prediction

    # Affichage du résultat de la prédiction
    if st.session_state.get('prediction_done', False):
        prediction = st.session_state['prediction']
        if prediction[0] == 0:
            st.write(f"Sol non adapté pour le maïs. Probabilité : {1 - prediction[1]}")
        else:
            st.success(f"Sol adapté pour le maïs. Probabilité : {prediction[1]}")

            # Entrée pour estimer le rendement
            st.subheader("Superficie plantée pour autres cultures (en km²)")
            corn_grain_area_km2 = st.number_input("Grain de mais", min_value=0.0, value=0.0)
            barley_area_km2 = st.number_input("Orge", min_value=0.0, value=0.0)
            sunflower_area_km2 = st.number_input("Tournesol", min_value=0.0, value=0.0)
            sugarbeet_area_km2 = st.number_input("Betterave à sucre", min_value=0.0, value=0.0)

            nouvelle_observation_dict_num = {
            "clay_0to30cm_percent": clay, 
            "silt_0to30cm_percent": silt, 
            "sand_0to30cm_percent": sand, 
            "ph_h2o_0to30cm": ph, 
            "organic_carbon_0to30cm_percent": organic_carbon, 
            "bdod_0to30cm": bdod, 
            "cfvo_0to30cm_percent": cfvo,
            "corn_grain_area_km2":corn_grain_area_km2,
            "barley_area_km2":barley_area_km2,
            "sunflower_area_km2":sunflower_area_km2,
            "sugarbeet_area_km2":sugarbeet_area_km2
            }

            if st.button("Estimer les rendements en km²"):
                nouvelle_observation_num = pd.DataFrame([nouvelle_observation_dict_num])
                prediction_num = make_prediction_ble_tendre_num(nouvelle_observation_num)
                st.session_state['estimation_done'] = True
                st.session_state['prediction_num'] = prediction_num

    # Affichage de l'estimation du rendement
    if st.session_state.get('estimation_done', False):
        st.write(f"Estimation du rendement : {st.session_state['prediction_num']} km²")

    # Bouton pour recommencer
    if st.session_state.get('prediction_done', False) or st.session_state.get('estimation_done', False):
        if st.button("Refaire une prédiction"):
            st.session_state.clear()
            st.experimental_rerun()  



if choix=="Orge":
    st.header("Orge")
    st.write("Entrez les caractéristiques du sol. Ensuite, validez avec le bouton Prédire.\nL'algorithme va estimer si ce sol est adpaté à la culture de l'Orge ou pas.\nSi c'est le cas, l'algorithme pourra également estimer les rendements esomptés")
    # Entrées utilisateur
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Paramètres physiques du sol à 30 cm de profondeur")
        clay = st.number_input("Teneur en argile", min_value=0.0, max_value=100.0, value=0.0)
        silt = st.number_input("Teneur de limon", min_value=0.0, max_value=100.0, value=0.0)
        sand = st.number_input("Teneur en sable", min_value=0.0, max_value=100.0, value=0.0)
        ph = st.number_input("pH du sol", min_value=0.0, max_value=14.0, value=7.0)
        organic_carbon = st.number_input("Teneur en carbone organique:", min_value=0.0, max_value=100.0, value=0.0)
        bdod = st.number_input("Densité apparente", min_value=0.0, max_value=5.0, value=0.0)
        cfvo = st.number_input("Matière organique", min_value=0.0, max_value=100.0, value=0.0)
    with col2:
        st.subheader("Présence d'autres cultures sur le sol")
        grain_mais = st.radio("Grain de mais:", ["Oui", "Non"], index=1,horizontal=True)
        ble_tendre = st.radio("Ble tendre:", ["Oui", "Non"], index=1,horizontal=True)
        tournsol = st.radio("Tournesol:", ["Oui", "Non"], index=1,horizontal=True)
        bettrave_a_sucre = st.radio("Bettrave à sucre:", ["Oui", "Non"], index=1,horizontal=True)
    nouvelle_observation_dict = {
            "clay_0to30cm_percent": clay, 
            "silt_0to30cm_percent": silt, 
            "sand_0to30cm_percent": sand, 
            "ph_h2o_0to30cm": ph, 
            "organic_carbon_0to30cm_percent": organic_carbon, 
            "bdod_0to30cm": bdod, 
            "cfvo_0to30cm_percent": cfvo,
            "grain_mais": 1 if grain_mais == "Oui" else 0, 
            "ble_tendre": 1 if ble_tendre == "Oui" else 0, 
            "tournsol": 1 if tournsol == "Oui" else 0, 
            "bettrave_a_sucre": 1 if bettrave_a_sucre == "Oui" else 0
        }
    nouvelle_observation = pd.DataFrame([nouvelle_observation_dict])

    # Bouton de prédiction
    if st.button("Prédire"):
        st.session_state['prediction_done'] = True
        st.session_state['estimation_done'] = False
        prediction = make_prediction_orge(nouvelle_observation)
        st.session_state['prediction'] = prediction

    # Affichage du résultat de la prédiction
    if st.session_state.get('prediction_done', False):
        prediction = st.session_state['prediction']
        if prediction[0] == 0:
            st.write(f"Sol non adapté pour le maïs. Probabilité : {1 - prediction[1]}")
        else:
            st.success(f"Sol adapté pour le maïs. Probabilité : {prediction[1]}")

            # Entrée pour estimer le rendement
            st.subheader("Superficie plantée pour autres cultures (en km²)")
            corn_grain_area_km2 = st.number_input("Grain de mais", min_value=0.0, value=0.0)
            soft_wheat_area_km2 = st.number_input("Ble tendre", min_value=0.0, value=0.0)
            sunflower_area_km2 = st.number_input("Tournesol", min_value=0.0, value=0.0)
            sugarbeet_area_km2 = st.number_input("Betterave à sucre", min_value=0.0, value=0.0)

            nouvelle_observation_dict_num = {
            "clay_0to30cm_percent": clay, 
            "silt_0to30cm_percent": silt, 
            "sand_0to30cm_percent": sand, 
            "ph_h2o_0to30cm": ph, 
            "organic_carbon_0to30cm_percent": organic_carbon, 
            "bdod_0to30cm": bdod, 
            "cfvo_0to30cm_percent": cfvo,
            "soft_wheat_area_km2":soft_wheat_area_km2,
            "corn_grain_area_km2":corn_grain_area_km2,
            "sunflower_area_km2":sunflower_area_km2,
            "sugarbeet_area_km2":sugarbeet_area_km2
            }

            if st.button("Estimer les rendements en km²"):
                nouvelle_observation_num = pd.DataFrame([nouvelle_observation_dict_num])
                prediction_num = make_prediction_orge_num(nouvelle_observation_num)
                st.session_state['estimation_done'] = True
                st.session_state['prediction_num'] = prediction_num

    # Affichage de l'estimation du rendement
    if st.session_state.get('estimation_done', False):
        st.write(f"Estimation du rendement : {st.session_state['prediction_num']} km²")

    # Bouton pour recommencer
    if st.session_state.get('prediction_done', False) or st.session_state.get('estimation_done', False):
        if st.button("Refaire une prédiction"):
            st.session_state.clear()
            st.experimental_rerun()  



if choix=="Bettrave à sucre":
    st.header("Bettrave à sucre")
    st.write("Entrez les caractéristiques du sol. Ensuite, validez avec le bouton Prédire.\nL'algorithme va estimer si ce sol est adpaté à la culture de la Bettrave à sucre ou pas.\nSi c'est le cas, l'algorithme pourra également estimer les rendements esomptés")
    
    # Entrées utilisateur
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Paramètres physiques du sol à 30 cm de profondeur")
        clay = st.number_input("Teneur en argile", min_value=0.0, max_value=100.0, value=0.0)
        silt = st.number_input("Teneur de limon", min_value=0.0, max_value=100.0, value=0.0)
        sand = st.number_input("Teneur en sable", min_value=0.0, max_value=100.0, value=0.0)
        ph = st.number_input("pH du sol", min_value=0.0, max_value=14.0, value=7.0)
        organic_carbon = st.number_input("Teneur en carbone organique:", min_value=0.0, max_value=100.0, value=0.0)
        bdod = st.number_input("Densité apparente", min_value=0.0, max_value=5.0, value=0.0)
        cfvo = st.number_input("Matière organique", min_value=0.0, max_value=100.0, value=0.0)
    with col2:
        st.subheader("Présence d'autres cultures sur le sol")
        grain_mais = st.radio("Grain de mais:", ["Oui", "Non"], index=1,horizontal=True)
        ble_tendre = st.radio("Ble tendre:", ["Oui", "Non"], index=1,horizontal=True)
        orge = st.radio("Orge:", ["Oui", "Non"], index=1,horizontal=True)
        tournsol = st.radio("Tournesol:", ["Oui", "Non"], index=1,horizontal=True)
    nouvelle_observation_dict = {
            "clay_0to30cm_percent": clay, 
            "silt_0to30cm_percent": silt, 
            "sand_0to30cm_percent": sand, 
            "ph_h2o_0to30cm": ph, 
            "organic_carbon_0to30cm_percent": organic_carbon, 
            "bdod_0to30cm": bdod, 
            "cfvo_0to30cm_percent": cfvo,
            "grain_mais": 1 if grain_mais == "Oui" else 0, 
            "ble_tendre": 1 if ble_tendre == "Oui" else 0, 
            "orge": 1 if orge == "Oui" else 0,
            "tournsol": 1 if tournsol == "Oui" else 0
        }
    nouvelle_observation = pd.DataFrame([nouvelle_observation_dict])

    # Bouton de prédiction
    if st.button("Prédire"):
        st.session_state['prediction_done'] = True
        st.session_state['estimation_done'] = False
        prediction = make_prediction_bettrave(nouvelle_observation)
        st.session_state['prediction'] = prediction

    # Affichage du résultat de la prédiction
    if st.session_state.get('prediction_done', False):
        prediction = st.session_state['prediction']
        if prediction[0] == 0:
            st.write(f"Sol non adapté pour le maïs. Probabilité : {1 - prediction[1]}")
        else:
            st.success(f"Sol adapté pour le maïs. Probabilité : {prediction[1]}")

            # Entrée pour estimer le rendement
            st.subheader("Superficie plantée pour autres cultures (en km²)")
            corn_grain_area_km2 = st.number_input("Grain de mais", min_value=0.0, value=0.0)
            soft_wheat_area_km2 = st.number_input("Ble tendre", min_value=0.0, value=0.0)
            sunflower_area_km2 = st.number_input("Tournesol", min_value=0.0, value=0.0)
            sugarbeet_area_km2 = st.number_input("Betterave à sucre", min_value=0.0, value=0.0)

            nouvelle_observation_dict_num = {
            "clay_0to30cm_percent": clay, 
            "silt_0to30cm_percent": silt, 
            "sand_0to30cm_percent": sand, 
            "ph_h2o_0to30cm": ph, 
            "organic_carbon_0to30cm_percent": organic_carbon, 
            "bdod_0to30cm": bdod, 
            "cfvo_0to30cm_percent": cfvo,
            "soft_wheat_area_km2":soft_wheat_area_km2,
            "corn_grain_area_km2":corn_grain_area_km2,
            "barley_area_km2":barley_area_km2,
            "sunflower_area_km2":sunflower_area_km2
            }

            if st.button("Estimer les rendements en km²"):
                nouvelle_observation_num = pd.DataFrame([nouvelle_observation_dict_num])
                prediction_num = make_prediction_bettrave_num(nouvelle_observation_num)
                st.session_state['estimation_done'] = True
                st.session_state['prediction_num'] = prediction_num

    # Affichage de l'estimation du rendement
    if st.session_state.get('estimation_done', False):
        st.write(f"Estimation du rendement : {st.session_state['prediction_num']} km²")

    # Bouton pour recommencer
    if st.session_state.get('prediction_done', False) or st.session_state.get('estimation_done', False):
        if st.button("Refaire une prédiction"):
            st.session_state.clear()
            st.experimental_rerun()




if choix == "Maïs":
    st.header("Maïs")
    st.write("Entrez les caractéristiques du sol. Ensuite, validez avec le bouton Prédire.\nL'algorithme va estimer si ce sol est adpaté à la culture du maïs ou pas.\nSi c'est le cas, l'algorithme pourra également estimer les rendements esomptés")

    # Entrées utilisateur
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Paramètres physiques du sol à 30 cm de profondeur")
        clay = st.number_input("Teneur en argile", min_value=0.0, max_value=100.0, value=0.0)
        silt = st.number_input("Teneur de limon", min_value=0.0, max_value=100.0, value=0.0)
        sand = st.number_input("Teneur en sable", min_value=0.0, max_value=100.0, value=0.0)
        ph = st.number_input("pH du sol", min_value=0.0, max_value=14.0, value=7.0)
        organic_carbon = st.number_input("Teneur en carbone organique:", min_value=0.0, max_value=100.0, value=0.0)
        bdod = st.number_input("Densité apparente", min_value=0.0, max_value=5.0, value=0.0)
        cfvo = st.number_input("Matière organique", min_value=0.0, max_value=100.0, value=0.0)
    with col2:
        st.subheader("Présence d'autres cultures sur le sol")
        ble_tendre = st.radio("Blé tendre:", ["Oui", "Non"], index=1,horizontal=True)
        orge = st.radio("Orge:", ["Oui", "Non"], index=1,horizontal=True)
        tournsol = st.radio("Tournesol:", ["Oui", "Non"], index=1,horizontal=True)
        bettrave_a_sucre = st.radio("Bettrave à sucre:", ["Oui", "Non"], index=1,horizontal=True)
        #st.success(f"Valeurs enregistrées ✅\n\nClay: {clay}, pH: {ph}, Blé tendre: {ble_tendre}")
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
    nouvelle_observation = pd.DataFrame([nouvelle_observation_dict])

    # Bouton de prédiction
    if st.button("Prédire"):
        st.session_state['prediction_done'] = True
        st.session_state['estimation_done'] = False
        prediction = make_prediction_grain_mais(nouvelle_observation)
        st.session_state['prediction'] = prediction

    # Affichage du résultat de la prédiction
    if st.session_state.get('prediction_done', False):
        prediction = st.session_state['prediction']
        if prediction[0] == 0:
            st.write(f"Sol non adapté pour le maïs. Probabilité : {1 - prediction[1]}")
        else:
            st.success(f"Sol adapté pour le maïs. Probabilité : {prediction[1]}")

            # Entrée pour estimer le rendement
            st.subheader("Superficie plantée pour autres cultures (en km²)")
            soft_wheat_area_km2 = st.number_input("Blé tendre", min_value=0.0, value=0.0)
            barley_area_km2 = st.number_input("Orge", min_value=0.0, value=0.0)
            sunflower_area_km2 = st.number_input("Tournesol", min_value=0.0, value=0.0)
            sugarbeet_area_km2 = st.number_input("Betterave à sucre", min_value=0.0, value=0.0)

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

            if st.button("Estimer les rendements en km²"):
                nouvelle_observation_num = pd.DataFrame([nouvelle_observation_dict_num])
                prediction_num = make_prediction_grain_mais_num(nouvelle_observation_num)
                st.session_state['estimation_done'] = True
                st.session_state['prediction_num'] = prediction_num

    # Affichage de l'estimation du rendement
    if st.session_state.get('estimation_done', False):
        st.write(f"Estimation du rendement : {st.session_state['prediction_num']} tonnes/km²")

    # Bouton pour recommencer
    if st.session_state.get('prediction_done', False) or st.session_state.get('estimation_done', False):
        if st.button("Refaire une prédiction"):
            st.session_state.clear()
            st.experimental_rerun()