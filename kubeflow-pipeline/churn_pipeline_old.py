# churn_pipeline.py
from typing import NamedTuple
from kfp import dsl
from kfp import compiler

# Point this at your MLflow tracking service (inside cluster DNS shown as example)
MLFLOW_URI = "http://mlflow.kflow-mlops.svc.cluster.local:5000"


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "great_expectations==0.18.12",
        "scikit-learn==1.5.1",
        "numpy==1.26.4",
    ],
)
def data_validation_op() -> str:
    """
    Creates a small synthetic 'churn' dataset and validates a simple data contract.
    On failure, raises -> pipeline stops.
    Returns the CSV path for downstream steps.
    """
    import pandas as pd
    from sklearn.datasets import make_classification
    import great_expectations as gx
    import os, json, tempfile

    X, y = make_classification(
        n_samples=1500, n_features=10, n_informative=6, random_state=42
    )
    cols = [f"f{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=cols)
    df["churn"] = y

    out_dir = tempfile.mkdtemp()
    csv_path = os.path.join(out_dir, "train.csv")
    df.to_csv(csv_path, index=False)

    # GE in-memory checkpoint with minimal expectations (column set + ranges)
    context = gx.get_context(mode="ephemeral")
    ds = context.sources.add_pandas(csv_path, name="train_df")
    ge_ds = ds.add_dataframe_asset(name="train_asset")
    batch = ge_ds.get_batch()

    # Expectations
    suite = context.add_expectation_suite("churn_contract")
    batch.expect_table_columns_to_match_set(set(cols + ["churn"]))
    for c in cols:
        batch.expect_column_values_to_not_be_null(c)
        batch.expect_column_median_to_be_between(c, min_value=-10, max_value=10)
    batch.expect_column_values_to_be_in_set("churn", [0, 1])

    results = context.assistants.validation.run(
        batch=batch, expectation_suite=suite, run_name="contract_check"
    )
    if not results["success"]:
        raise ValueError("Data contract failed: " + json.dumps(results, default=str))

    return csv_path


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "mlflow==2.14.1",
        "scikit-learn==1.5.1",
        "xgboost==2.1.1",
        "numpy==1.26.4",
    ],
)
def train_and_log_op(csv_path: str) -> NamedTuple("Outputs", [("run_id", str), ("model_uri", str)]):
    """
    Trains XGBoost, logs to MLflow, returns (run_id, model_uri).
    """
    import os, pandas as pd, mlflow
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7)

    with mlflow.start_run(run_name="xgb_churn") as run:
        params = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="auc",
        )
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_proba)

        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", float(auc))
        mlflow.log_metric("n_test", int(len(y_te)))

        # Log model
        mlflow.xgboost.log_model(model, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")

        # Try registering (ok if registry not enabled in local demo)
        try:
            mlflow.register_model(model_uri, "churn_xgb")
        except Exception:
            pass

        return (run.info.run_id, model_uri)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "mlflow==2.14.1",
        "evidently==0.4.15",
        "scikit-learn==1.5.1",
        "numpy==1.26.4",
    ],
)
def drift_check_op(csv_path: str, run_id: str) -> str:
    """
    Generates an Evidently data drift HTML and logs it to MLflow artifacts under the same run.
    Returns path to the HTML file (for viewer components if needed).
    """
    import os, pandas as pd, tempfile, mlflow
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI

    df = pd.read_csv(csv_path)
    # Simulate "current" data by nudging a couple features
    current = df.copy()
    for c in ["f0", "f1"]:
        current[c] = current[c] * 1.1 + 0.1

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=df.drop(columns=["churn"]),
        current_data=current.drop(columns=["churn"]),
    )

    tmp = tempfile.mkdtemp()
    html_path = os.path.join(tmp, "drift_report.html")
    report.save_html(html_path)

    # Attach the report to the same MLflow run
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_artifact(html_path, artifact_path="evidently")

    return html_path


@dsl.pipeline(name="churn-mlops-pipeline")
def churn_pipeline():
    v = data_validation_op()
    t = train_and_log_op(csv_path=v.output)
    _ = drift_check_op(csv_path=v.output, run_id=t.outputs["run_id"])


if __name__ == "__main__":
    compiler.Compiler().compile(churn_pipeline, package_path="churn_pipeline.json")

