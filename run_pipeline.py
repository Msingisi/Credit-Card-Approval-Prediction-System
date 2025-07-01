from pipelines.training_pipeline import train_pipeline
from steps.config import ModelNameConfig
from zenml.client import Client

def main():
    data_path = "clean_data/cleaned_credit_data.csv"
    all_models = ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost", "MLP"]

    print(f"MLflow Tracking URI: {Client().active_stack.experiment_tracker.get_tracking_uri()}")

    for model_name in all_models:
        print(f"\nTraining model: {model_name}")
        config = ModelNameConfig(model_name=model_name)
        try:
            train_pipeline(data_path=data_path, config=config)
            print(f"Finished training {model_name}")
        except Exception as e:
            print(f"Error training {model_name}: {e}")

if __name__ == "__main__":
    main()