import os
import platform
import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from pipelines.deployment_pipeline import continuous_deployment_pipeline

DEPLOY = "deploy"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose whether to deploy the model (`deploy`) or both (`deploy_and_predict`). Prediction pipeline is deprecated.",
)
@click.option(
    "--min-accuracy",
    default=0.80,
    help="Minimum accuracy required to deploy the model.",
)
def run_deployment(config: str, min_accuracy: float):
    """Run the ZenML deployment pipeline with MLflow integration."""
    deploy = config in [DEPLOY, DEPLOY_AND_PREDICT]

    if deploy:
        print("[bold yellow] Running deployment pipeline...[/bold yellow]")
        continuous_deployment_pipeline(
            data_path="clean_data/cleaned_credit_data.csv",
            min_accuracy=min_accuracy,
            workers=1,
            timeout=60,
        )

        if platform.system() == "Windows":
            print("\n[bold red] Daemon-based model serving is not supported on Windows.[/bold red]")
            print("[bold yellow] Your pipeline ran successfully and the model met the accuracy threshold.[/bold yellow]")
            print("[bold cyan] Please manually serve your model using MLflow CLI.[/bold cyan]")
            print("\nExample:")
            print("[green]mlflow models serve --model-uri \"runs:/<your_run_id>/model\" --port 5005 --env-manager=local\"[/green]")
            print("💡 Replace `<your_run_id>` with the actual MLflow run ID shown in the MLflow UI.")
            print("You can inspect runs using:\n"
                  f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]\n")
            return
        else:
            model_deployer = MLFlowModelDeployer.get_active_model_deployer()
            services = model_deployer.find_model_server(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step"
            )

            if not services:
                print("[bold red] No model server found. Deployment may have failed or was skipped.[/bold red]")
                return

            model_uri = services[0].predictor.model_uri
            print(f"[bold green] Deploying MLflow model at {model_uri}[/bold green]")
            os.system(f'mlflow models serve --model-uri "{model_uri}" --port 5000')

    print(
        "You can now run:\n"
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]\n"
        "...to inspect experiment runs."
    )


if __name__ == "__main__":
    run_deployment()