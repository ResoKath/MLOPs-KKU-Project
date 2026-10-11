import mlflow
import pandas as pd
import sys
import os

def validate_data(raw_data_path="data/Zomato Dataset.csv", output_clean_path="data/staging/validated_data.csv"):
    # Set SQLite tracking URI and experiment name
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Food_Delivery_MLOps")
    
    with mlflow.start_run(run_name="01_Data_Validation"):
        print(f"Loading raw data from: {raw_data_path}...")
        df = pd.read_csv(raw_data_path)
        
        # 1. Calculate & log basic dataset statistics
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_values_count = int(df.isnull().sum().sum())
        missing_rate = float(missing_values_count / (total_rows * total_cols))
        
        # Log metrics to MLflow
        mlflow.log_metric("total_rows", total_rows)
        mlflow.log_metric("total_columns", total_cols)
        mlflow.log_metric("missing_values_count", missing_values_count)
        mlflow.log_metric("missing_rate", missing_rate)
        
        print(f"Dataset Loaded: {total_rows} rows, {total_cols} columns, missing rate: {missing_rate:.2%}")
        
        # 2. Check schema & rules (Quality Gate)
        required_columns = [
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Weather_conditions",
            "Road_traffic_density",
            "Vehicle_condition",
            "Type_of_order",
            "Type_of_vehicle",
            "Time_taken (min)"
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            mlflow.log_param("validation_status", "FAILED")
            raise ValueError(f"Data Validation Failed! Missing required columns: {missing_cols}")
        
        if missing_rate > 0.20:  # Threshold: Fail if more than 20% total cells are missing
            mlflow.log_param("validation_status", "FAILED")
            raise ValueError(f"Data Validation Failed! Too many missing values ({missing_rate:.2%}).")
        
        # Target variable sanity check
        if (df["Time_taken (min)"] <= 0).any():
            mlflow.log_param("validation_status", "FAILED")
            raise ValueError("Data Validation Failed! Found non-positive delivery times.")
        
        # 3. Log passing status
        mlflow.log_param("validation_status", "PASSED")
        print("Data validation passed successfully!")
        
        # 4. Save validated data for Script 02
        os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)
        df.to_csv(output_clean_path, index=False)
        print(f"Validated data saved to: {output_clean_path}")
        
        # 5. Save validation report artifact to MLflow
        report_path = "validation_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Status: PASSED\n")
            f.write(f"Rows: {total_rows}\n")
            f.write(f"Cols: {total_cols}\n")
            f.write(f"Missing Values: {missing_values_count}\n")
            f.write(f"Missing Rate: {missing_rate:.4f}\n")
        mlflow.log_artifact(report_path)

if __name__ == "__main__":
    validate_data()