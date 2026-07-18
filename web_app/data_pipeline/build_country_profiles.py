import os
import json
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    master_csv = os.path.join(base_dir, "../../data/processed/master_dataset_hourly.csv")
    output_json = os.path.join(base_dir, "country_profiles.json")

    print("Loading master dataset...")
    df = pd.read_csv(master_csv, usecols=["Country", "Load", "Renewables_MW", "Real_Wholesale_Price_EUR"])
    
    # Calculate Residual Load
    df["Residual_Load"] = df["Load"] - df["Renewables_MW"]

    print("Calculating Continuous Profiles for 17 countries...")
    profiles = {}
    
    for country, grp in df.groupby("Country"):
        avg_load = grp["Load"].mean()
        avg_res_load = grp["Residual_Load"].mean()
        avg_price = grp["Real_Wholesale_Price_EUR"].mean()
        
        profiles[country] = {
            "Country_Avg_Load": round(float(avg_load), 4),
            "Country_Avg_Residual_Load": round(float(avg_res_load), 4),
            "Country_Avg_Price": round(float(avg_price), 4)
        }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=4)
    
    print(f"Successfully generated {len(profiles)} country profiles -> {output_json}")

if __name__ == "__main__":
    main()
