"""
compare_ols_features.py
Thu nghiem: OLS voi 18 bien (SAFE_FEATURES) vs 6 bien (C1_FEATURES).
Khong sua baseline.py hay forecaster.py.
Chay: python src/experiments/compare_ols_features.py
"""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

C1_FEATURES = [
    "Residual_Load", "TTF_Gas_Price", "Coal_Price",
    "EU_ETS_Price", "Brent_Oil_Price", "EU_Gas_Storage_Anomaly",
]

SAFE_FEATURES = [
    "Load", "Residual_Load", "Net_Import",
    "TTF_Gas_Price", "Coal_Price", "EU_ETS_Price",
    "Brent_Oil_Price", "EU_Gas_Storage_Anomaly",
    "Wind_Onshore_MW", "Wind_Offshore_MW", "Solar_MW",
    "Biomass_MW", "Geothermal_MW", "Hydro_RoR_MW", "Hydro_Reservoir_MW",
    "Nuclear_MW", "Hour_Sin", "Hour_Cos",
]

TARGET = "Real_Wholesale_Price_EUR"
COUNTRIES = ["DE", "DK", "ES", "FR", "IT", "PL"]


def ols_r2(df, features):
    avail = [f for f in features if f in df.columns]
    df2 = df.dropna(subset=avail + [TARGET])
    X, y = df2[avail].values, df2[TARGET].values
    return round(float(r2_score(y, LinearRegression().fit(X, y).predict(X))), 4)


def vif_table(df, features):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    avail = [f for f in features if f in df.columns]
    df2 = df.dropna(subset=avail)
    X = sm.add_constant(df2[avail])
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const": continue
        try:   v = round(variance_inflation_factor(X.values, i), 1)
        except: v = float("nan")
        rows.append({"Feature": col, "VIF": v})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def main():
    # --- Load data ---
    for path in ["data/processed/features_hourly.parquet",
                 "data/processed/features_hourly.csv"]:
        try:
            df = pd.read_parquet(path) if path.endswith(".parquet") \
                 else pd.read_csv(path, parse_dates=["datetime"])
            print(f"[OK] Loaded: {path}  shape={df.shape}")
            break
        except Exception as e:
            print(f"[skip] {path}: {e}")
    else:
        print("[ERROR] Khong tim thay file du lieu."); sys.exit(1)

    # --- R2 so sanh ---
    print("\n" + "="*60)
    print("SO SANH R2 OLS: 18 bien vs 6 bien C1")
    print("="*60)
    rows = []
    for c in COUNTRIES:
        dfc = df[df["Country"] == c].copy()
        r18 = ols_r2(dfc, SAFE_FEATURES)
        r6  = ols_r2(dfc, C1_FEATURES)
        d   = round(r18 - r6, 4)
        note = "18 tot hon" if d > 0.01 else ("= nhau" if abs(d) <= 0.01 else "6 tot hon")
        rows.append({"Country": c, "R2_18var": r18, "R2_6var": r6, "Delta(18-6)": d, "Nhan xet": note})
        print(f"  {c}: R2_18={r18:+.4f}  R2_6={r6:+.4f}  delta={d:+.4f}  [{note}]")

    result = pd.DataFrame(rows)
    avg = result["Delta(18-6)"].mean()
    print(f"\n  Trung binh delta = {avg:+.4f}")

    # --- VIF C1 (DE) ---
    print("\n" + "-"*60)
    print("VIF -- 6 bien C1 (DE):")
    df_de = df[df["Country"] == "DE"]
    print(vif_table(df_de, C1_FEATURES).to_string(index=False))

    # --- VIF SAFE (DE) ---
    print("\nVIF -- 18 bien SAFE (DE):")
    print(vif_table(df_de, SAFE_FEATURES).to_string(index=False))
    print("\nNguong: VIF<5 tot | VIF 5-10 chap nhan | VIF>10 rui ro | VIF=inf exact collinearity")

    # --- Ket luan ---
    print("\n" + "="*60)
    print("KET LUAN:")
    if avg > 0.02:
        print("  18 bien cho R2 cao hon dang ke.")
        print("  Kiem tra VIF -- neu cao, R2 bi inflate.")
    elif abs(avg) <= 0.02:
        print("  R2 hai cau hinh gan bang nhau.")
        print("  -> 6 bien C1 la du, nhat quan voi XGBoost/DAG.")
    else:
        print("  6 bien C1 tot hon -- them bien gay collinearity/overfit.")

if __name__ == "__main__":
    main()
