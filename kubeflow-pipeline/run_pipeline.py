# run_pipeline.py
from kfp import Client

# If you're port-forwarding the UI:
# kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80
HOST = "http://localhost:8080/pipeline"

client = Client(host=HOST)

experiment = client.create_experiment(name="Churn MLOps Demo")
run = client.create_run_from_pipeline_package(
    pipeline_file="churn_pipeline.json",
    arguments={},  # no params
    experiment_id=experiment.id,
    run_name="churn-mlops-run",
)

print("Run created:", run.run_id)
print("UI:", f"{HOST}/#/runs/details/{run.run_id}")

