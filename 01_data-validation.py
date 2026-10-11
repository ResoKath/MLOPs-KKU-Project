import mlflow
import pandas as pd
import sys
import os

def validate_data(raw_data_path, output_clean_path):
    # Set the experiment name in MLflow
    mlflow.set_experiment("Data_Validation_Pipeline")
    
    with mlflow.start_run(run_name="01_Data_Validation"):
        print("Loading raw data...")
        df = pd.read_csv(raw_data_path)
        
        # 1. Calculate & log basic dataset statistics
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_values_count = int(df.isnull().sum().sum())
        missing_rate = missing_values_count / (total_rows * total_cols)
        
        # Log metrics to MLflow
        mlflow.log_metric("total_rows", total_rows)
        mlflow.log_metric("total_columns", total_cols)
        mlflow.log_metric("missing_values_count", missing_values_count)
        mlflow.log_metric("missing_rate", missing_rate)
        
        print(f"Dataset Loaded: {total_rows} rows, {total_cols} columns.")
        
        # 2. Check schema & rules (Quality Gate)
        required_columns = ["age", "income", "churn"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            mlflow.log_param("validation_status", "FAILED")
            raise ValueError(f"Data Validation Failed! Missing required columns: {missing_cols}")
        
        if missing_rate > 0.30:  # Threshold: Fail if more than 30% data missing
            mlflow.log_param("validation_status", "FAILED")
            raise ValueError("Data Validation Failed! Too many missing values.")
        
        # 3. Log passing status
        mlflow.log_param("validation_status", "PASSED")
        print("Data validation passed successfully!")
        
        # 4. Save cleaned/validated data for Script 02
        os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)
        df.to_csv(output_clean_path, index=False)
        
        # Optional: Save a validation report artifact to MLflow
        report_path = "validation_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Status: PASSED\nRows: {total_rows}\nCols: {total_cols}\n")
        mlflow.log_artifact(report_path)

if __name__ == "__main__":
    validate_data("data/raw/data.csv", "data/staging/validated_data.csv")