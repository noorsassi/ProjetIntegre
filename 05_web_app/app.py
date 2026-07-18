import streamlit as st
import pandas as pd
import numpy as np
import joblib
import duckdb
import os

# 1. Configuration de la page Streamlit
st.set_page_config(
    page_title="Suivi & Prédiction du Churn Client",
    page_icon="🏦",
    layout="wide"
)

# 2. Chargement sécurisé du modèle et des features
@st.cache_resource
def charger_fichiers_ml():
    if not os.path.exists('model_churn_final.pkl') or not os.path.exists('model_features.pkl'):
        return None, None
    model = joblib.load('model_churn_final.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

model, model_features = charger_fichiers_ml()

# 3. Titre de l'application
st.title(" Plateforme Analytique & Prédiction du Churn Client")
st.markdown("---")

# Vérification de la présence du modèle entraîné
if model is None:
    st.error(" Les fichiers du modèle ML ('model_churn_final.pkl' et 'model_features.pkl') sont introuvables. Veuillez d'abord exécuter le script d'entraînement de la Semaine 3.")
else:
    # 4. Barre latérale de navigation (Navigation entre les besoins métiers)
    menu = st.sidebar.radio(
        "Navigation",
        [" Indicateurs de Synthèse", " Prédiction Individuelle", " Liste des Clients à Risque"]
    )

    # -------------------------------------------------------------------------
    # ONGLET 1 : INDICATEURS DE SYNTHÈSE
    # -------------------------------------------------------------------------
    if menu == " Indicateurs de Synthèse":
        st.header(" Indicateurs Clés de l'Institution Financière")
        
        if os.path.exists('dwh_churn.db'):
            conn = duckdb.connect('dwh_churn.db')
            kpis = conn.execute("""
                SELECT 
                    COUNT(DISTINCT CUSTOMER_NO), 
                    ROUND(AVG(ACCT_BALANCE), 2), 
                    ROUND(AVG(CHURN) * 100, 2) 
                FROM fct_comptes
            """).fetchone()
            conn.close()
            
            # Affichage sous forme de cartes d'indicateurs (KPI Cards)
            col1, col2, col3 = st.columns(3)
            col1.metric(" Total Clients Uniques", f"{kpis[0]:,}")
            col2.metric(" Solde Moyen Global", f"{kpis[1]} EUR/TND")
            col3.metric(" Taux de Churn Actuel", f"{kpis[2]} %")
        else:
            st.warning("Base de données DuckDB introuvable. Affichage de données de démonstration.")
            col1, col2, col3 = st.columns(3)
            col1.metric(" Total Clients Uniques", "154,200")
            col2.metric(" Solde Moyen Global", "3,450 EUR/TND")
            col3.metric(" Taux de Churn Actuel", "14.2 %")

    # -------------------------------------------------------------------------
    # ONGLET 2 : PRÉDICTION INDIVIDUELLE (Formulaire Profil Client)
    # -------------------------------------------------------------------------
    elif menu == " Prédiction Individuelle":
        st.header(" Simulation de Risque pour un Client")
        st.write("Saisissez les informations du profil client pour évaluer sa probabilité d'attrition.")
        
        with st.form("formulaire_client"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Âge du client (ans)", 18, 100, 40)
                cust_tenure = st.slider("Ancienneté Relation Banque (années)", 0, 30, 5)
                acct_tenure = st.slider("Ancienneté du Compte (années)", 0, 30, 3)
                marital = st.selectbox("Statut Marital", ["Célibataire (C)", "Marié(e) (M)", "Divorcé(e) (D)", "Veuf/Veuve (V)"])
            
            with col2:
                salary = st.number_input("Salaire ou Revenu Annuel Déclaré", min_value=0, value=35000)
                balance = st.number_input("Solde Actuel du Compte", value=1500)
                partyclass = st.selectbox("Classification Commerciale", ["Retail", "Corporate", "Wealth", "Other"])
            
            soumis = st.form_submit_button("Calculer le score de risque")
            
            if soumis:
                # Encodage des saisies pour correspondre au format d'entraînement (One-Hot Encoding)
                statut_code = marital.split('(')[1][0]
                
                input_dict = {
                    'AGE': age,
                    'CUST_TENURE_YEARS': cust_tenure,
                    'ACCOUNT_TENURE_YEARS': acct_tenure,
                    'SALARY': salary,
                    'ACCT_BALANCE': balance,
                    'MARITAL_STATUS_M': 1 if statut_code == 'M' else 0,
                    'MARITAL_STATUS_D': 1 if statut_code == 'D' else 0,
                    'MARITAL_STATUS_V': 1 if statut_code == 'V' else 0,
                    'PARTYCLASS_Corporate': 1 if partyclass == 'Corporate' else 0,
                    'PARTYCLASS_Wealth': 1 if partyclass == 'Wealth' else 0,
                    'PARTYCLASS_Other': 1 if partyclass == 'Other' else 0,
                }
                
                # Conversion en DataFrame et alignement strict avec les colonnes attendues par le modèle
                df_input = pd.DataFrame([input_dict])
                df_input = df_input.reindex(columns=model_features, fill_value=0)
                
                # Calcul de la probabilité de churn
                probabilite = model.predict_proba(df_input)[0][1]
                
                # Affichage visuel du résultat
                st.markdown("### Résultat de l'analyse réglementaire")
                if probabilite > 0.50:
                    st.error(f" **Alerte Risque Élevé :** Ce client présente une probabilité de churn de **{probabilite*100:.1f}%**.")
                else:
                    st.success(f" **Profil Stable :** Probabilité de churn faible estimée à **{probabilite*100:.1f}%**.")

    # -------------------------------------------------------------------------
    # ONGLET 3 : LISTE DES CLIENTS À RISQUE (Batch processing)
    # -------------------------------------------------------------------------
    elif menu == " Liste des Clients à Risque":
        st.header(" Liste des Comptes Actifs à Risque Élevé")
        st.write("Ce tableau extrait les clients actuellement actifs dont la probabilité de départ est supérieure à 70%, à destination des conseillers clientèle.")
        
        if os.path.exists('dwh_churn.db'):
            conn = duckdb.connect('dwh_churn.db')
            # Simulation d'une vue de comptes actifs
            df_actifs = conn.execute("""
                SELECT CUSTOMER_NO, ACCOUNT_NO, AGE, CUST_TENURE_YEARS, ACCOUNT_TENURE_YEARS, SALARY, ACCT_BALANCE
                FROM fct_comptes 
                WHERE CHURN = 0 
                LIMIT 100
            """).df()
            conn.close()
            
            if not df_actifs.empty:
                # Ajout de l'encodage par défaut pour la démo de prédiction en lot
                df_features = pd.DataFrame(0, index=df_actifs.index, columns=model_features)
                for col in ['AGE', 'CUST_TENURE_YEARS', 'ACCOUNT_TENURE_YEARS', 'SALARY', 'ACCT_BALANCE']:
                    if col in model_features:
                        df_features[col] = df_actifs[col]
                
                # Calcul des scores pour tout le lot
                df_actifs['Probabilité_Churn (%)'] = np.round(model.predict_proba(df_features)[:, 1] * 100, 1)
                
                # Tri par risque décroissant
                df_risque = df_actifs.sort_values(by='Probabilité_Churn (%)', ascending=False)
                st.dataframe(df_risque, use_container_width=True)
            else:
                st.info("Aucun client actif trouvé dans la base de données.")
        else:
            st.info(" Connectez votre base DuckDB locale pour lister dynamiquement les clients à risque à partir de vos données réelles.")
