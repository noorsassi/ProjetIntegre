import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

def entrainer_pipeline_ml():
    print(" [ML] Extraction des données d'entraînement depuis le DWH...")
    conn = duckdb.connect('dwh_churn.db')
    
    # Sélection des features analytiques identifiées lors de l'EDA
    query = """
        SELECT 
            AGE, CUST_TENURE_YEARS, ACCOUNT_TENURE_YEARS, 
            SALARY, ACCT_BALANCE, MARITAL_STATUS, PARTYCLASS, CHURN
        FROM fct_comptes
    """
    df_ml = conn.execute(query).df()
    conn.close()
    
    # 1. Encodage des variables catégorielles (One-Hot Encoding)
    df_ml = pd.get_dummies(df_ml, columns=['MARITAL_STATUS', 'PARTYCLASS'], drop_first=True)
    
    # Séparation Features (X) et Cible (y)
    X = df_ml.drop(columns=['CHURN'])
    y = df_ml['CHURN']
    
    # Sauvegarde de la liste exacte des colonnes requises pour l'application Web
    features_list = X.columns.tolist()
    joblib.dump(features_list, 'model_features.pkl')
    
    # 2. Split Train/Test Stratifié (Crucial pour conserver la proportion de churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Traitement du déséquilibre de classe par calcul du poids (scale_pos_weight)
    poids_classe_neg = sum(y_train == 0)
    poids_classe_pos = sum(y_train == 1)
    ratio_poids = poids_classe_neg / poids_classe_pos
    
    print(f" Ratio de déséquilibre des classes calculé : {ratio_poids:.2f}")
    
    # 4. Entraînement et comparaison des modèles
    print("\n Entraînement du modèle XGBoost avec gestion du déséquilibre...")
    model_xgb = XGBClassifier(
        scale_pos_weight=ratio_poids,
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    model_xgb.fit(X_train, y_train)
    
    # 5. Évaluation rigoureuse (Pas seulement l'accuracy !)
    y_pred = model_xgb.predict(X_test)
    y_proba = model_xgb.predict_proba(X_test)[:, 1]
    
    print("\n --- RAPPORT DE PERFORMANCE (Jeu de Test) ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score : {roc_auc_score(y_test, y_proba):.4f}")
    
    # 6. Sauvegarde du modèle finalisé pour l'interface web
    joblib.dump(model_xgb, 'model_churn_final.pkl')
    print("\n [SUCCÈS] Modèle de classification XGBoost sauvegardé sous 'model_churn_final.pkl'")

if __name__ == "__main__":
    entrainer_pipeline_ml()
