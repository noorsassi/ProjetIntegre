import duckdb
import os

def generer_donnees_powerbi():
    print("📊 [BI] Connexion au Data Warehouse DuckDB...")
    conn = duckdb.connect('dwh_churn.db')
    
    # 1. Validation et calcul des KPIs Globaux demandés par l'énoncé
    print("\n📈 --- CALCUL DES INDICATEURS MÉTIERS DE RÉFÉRENCE ---")
    
    kpis = conn.execute("""
        SELECT 
            COUNT(DISTINCT CUSTOMER_NO) as Total_Clients,
            COUNT(ACCOUNT_NO) as Total_Comptes,
            ROUND(AVG(AGE), 1) as Age_Moyen,
            ROUND(AVG(ACCT_BALANCE), 2) as Solde_Moyen,
            ROUND(AVG(CHURN) * 100, 2) as Taux_Churn_Global
        FROM fct_comptes
    """).fetchone()
    
    print(f" Nombre total de clients uniques : {kpis[0]}")
    print(f" Nombre total de comptes : {kpis[1]}")
    print(f" Âge moyen de la clientèle : {kpis[2]} ans")
    print(f" Solde moyen par compte : {kpis[3]} EUR/TND")
    print(f" Taux de Churn Global : {kpis[4]} %")
    
    # 2. Exportation des tables du modèle en étoile pour Power BI
    print("\n [Export] Génération des fichiers de données pour Power BI...")
    os.makedirs("export_powerbi", exist_ok=True)
    
    # Export de la table de faits et des dimensions au format Parquet
    tables_a_exporter = ['fct_comptes', 'dim_industry', 'dim_currency', 'dim_closure', 'dim_category']
    
    for table in tables_a_exporter:
        conn.execute(f"COPY {table} TO 'export_powerbi/{table}.parquet' (FORMAT PARQUET);")
        print(f" Table '{table}' exportée dans : export_powerbi/{table}.parquet")
        
    conn.close()
    print("\n  [SUCCÈS] Vos fichiers sont prêts. Dans Power BI Desktop, utilisez 'Obtenir les données' -> 'Parquet' pour charger ces fichiers.")

if __name__ == "__main__":
    generer_donnees_powerbi()
