"""
Churn Pipeline (Kubeflow Pipelines v2)
- Typed artifacts (Dataset, Model, Metrics)
- Synthetic data -> split -> train -> eval -> decision
- MLflow logs artifacts to MinIO (S3-compatible)

Compile:
    python churn_pipeline.py --compile   # -> churn_pipeline_v1.json
"""

import argparse
import sys
from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics, OutputPath


# --------------------
# Components
# --------------------

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "mlflow==2.14.1",
        "boto3==1.34.162",
    ],
)
def ingest_synthetic(
    output_dataset: Output[Dataset],
    mlflow_tracking_uri: str = "",
    s3_endpoint: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    random_state: int = 42,
    n_samples: int = 3000,
    n_features: int = 20,
    n_informative: int = 8,
    class_sep: float = 1.2,
):
    import os, mlflow
    import pandas as pd
    from sklearn.datasets import make_classification

    # Keep helpers INSIDE the component (KFP serializes only function body)
    def _setup_env(tracking, s3, key, secret, region="us-east-1"):
        def clean(v): return v.strip() if isinstance(v, str) else v
        tracking, s3, key, secret = map(clean, (tracking, s3, key, secret))
        if tracking: os.environ["MLFLOW_TRACKING_URI"] = tracking
        if s3:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3
            os.environ["AWS_ENDPOINT_URL_S3"] = s3
        if key:    os.environ["AWS_ACCESS_KEY_ID"] = key
        if secret: os.environ["AWS_SECRET_ACCESS_KEY"] = secret
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        os.environ.setdefault("AWS_S3_FORCE_PATH_STYLE", "true")

    _setup_env(mlflow_tracking_uri, s3_endpoint, aws_access_key_id, aws_secret_access_key)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        class_sep=class_sep,
        random_state=random_state,
    )
    cols = [f"f{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["churn"] = y.astype(int)

    os.makedirs(os.path.dirname(output_dataset.path), exist_ok=True)
    df.to_csv(output_dataset.path, index=False)

    mlflow.set_experiment("churn-kfp")
    with mlflow.start_run(run_name="ingest_synthetic"):
        mlflow.log_params({
            "random_state": random_state,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_informative": n_informative,
            "class_sep": class_sep,
        })
        mlflow.log_text(df.head(3).to_csv(index=False), artifact_file="preview.csv")


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "scikit-learn==1.4.2",
        "mlflow==2.14.1",
        "boto3==1.34.162",
    ],
)
def preprocess_split(
    raw_dataset: Input[Dataset],
    X_train_ds: Output[Dataset],
    X_val_ds: Output[Dataset],
    y_train_ds: Output[Dataset],
    y_val_ds: Output[Dataset],
    mlflow_tracking_uri: str = "",
    s3_endpoint: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    test_size: float = 0.2,
    random_state: int = 42,
):
    import os, mlflow
    import pandas as pd
    from sklearn.model_selection import train_test_split

    def _setup_env(tracking, s3, key, secret, region="us-east-1"):
        def clean(v): return v.strip() if isinstance(v, str) else v
        tracking, s3, key, secret = map(clean, (tracking, s3, key, secret))
        if tracking: os.environ["MLFLOW_TRACKING_URI"] = tracking
        if s3:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3
            os.environ["AWS_ENDPOINT_URL_S3"] = s3
        if key:    os.environ["AWS_ACCESS_KEY_ID"] = key
        if secret: os.environ["AWS_SECRET_ACCESS_KEY"] = secret
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        os.environ.setdefault("AWS_S3_FORCE_PATH_STYLE", "true")

    _setup_env(mlflow_tracking_uri, s3_endpoint, aws_access_key_id, aws_secret_access_key)

    df = pd.read_csv(raw_dataset.path)
    feature_cols = [c for c in df.columns if c != "churn"]
    X = df[feature_cols]
    y = df["churn"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs(os.path.dirname(X_train_ds.path), exist_ok=True)
    X_train.to_csv(X_train_ds.path, index=False)
    X_val.to_csv(X_val_ds.path, index=False)
    y_train.to_csv(y_train_ds.path, index=False, header=True)
    y_val.to_csv(y_val_ds.path, index=False, header=True)

    mlflow.set_experiment("churn-kfp")
    with mlflow.start_run(run_name="preprocess_split"):
        mlflow.log_params({
            "test_size": test_size,
            "random_state": random_state,
            "n_features": X.shape[1],
        })
        mlflow.log_metrics({
            "train_rows": float(len(X_train)),
            "val_rows": float(len(X_val)),
            "class_balance_train": float(y_train.mean()),
            "class_balance_val": float(y_val.mean()),
        })
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_list.txt")


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "mlflow==2.14.1",
        "joblib==1.3.2",
        "boto3==1.34.162",
    ],
)
def train_model(
    X_train_ds: Input[Dataset],
    y_train_ds: Input[Dataset],
    model_out: Output[Model],
    train_metrics: Output[Metrics],
    mlflow_tracking_uri: str = "",
    s3_endpoint: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    penalty: str = "l2",
    C: float = 1.0,
    max_iter: int = 500,
    registered_model_name: str = "churn-sklearn",
):
    import os, joblib, mlflow, mlflow.sklearn
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from mlflow.models.signature import infer_signature

    def _setup_env(tracking, s3, key, secret, region="us-east-1"):
        def clean(v): return v.strip() if isinstance(v, str) else v
        tracking, s3, key, secret = map(clean, (tracking, s3, key, secret))
        if tracking: os.environ["MLFLOW_TRACKING_URI"] = tracking
        if s3:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3
            os.environ["AWS_ENDPOINT_URL_S3"] = s3
        if key:    os.environ["AWS_ACCESS_KEY_ID"] = key
        if secret: os.environ["AWS_SECRET_ACCESS_KEY"] = secret
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        os.environ.setdefault("AWS_S3_FORCE_PATH_STYLE", "true")

    _setup_env(mlflow_tracking_uri, s3_endpoint, aws_access_key_id, aws_secret_access_key)

    X_train = pd.read_csv(X_train_ds.path)
    y_train = pd.read_csv(y_train_ds.path).iloc[:, 0].astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty=penalty, C=C, max_iter=max_iter, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)

    auc_train = float(roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1]))

    os.makedirs(os.path.dirname(model_out.path), exist_ok=True)
    joblib.dump(pipe, model_out.path)

    train_metrics.log_metric("auc_train", auc_train)

    mlflow.set_experiment("churn-kfp")
    with mlflow.start_run(run_name="train_model"):
        mlflow.log_params({"penalty": penalty, "C": C, "max_iter": max_iter})
        mlflow.log_metrics({"auc_train": auc_train})
        signature = infer_signature(
            X_train.head(32), pipe.predict_proba(X_train.head(32))[:, 1]
        )
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=registered_model_name,
        )


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "mlflow==2.14.1",
        "joblib==1.3.2",
        "boto3==1.34.162",
    ],
)
def evaluate_model(
    model_in: Input[Model],
    X_val_ds: Input[Dataset],
    y_val_ds: Input[Dataset],
    eval_metrics: Output[Metrics],
    auc_value: OutputPath(float),
    mlflow_tracking_uri: str = "",
    s3_endpoint: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
):
    import os, json, joblib, mlflow
    import pandas as pd
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

    def _setup_env(tracking, s3, key, secret, region="us-east-1"):
        def clean(v): return v.strip() if isinstance(v, str) else v
        tracking, s3, key, secret = map(clean, (tracking, s3, key, secret))
        if tracking: os.environ["MLFLOW_TRACKING_URI"] = tracking
        if s3:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3
            os.environ["AWS_ENDPOINT_URL_S3"] = s3
        if key:    os.environ["AWS_ACCESS_KEY_ID"] = key
        if secret: os.environ["AWS_SECRET_ACCESS_KEY"] = secret
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        os.environ.setdefault("AWS_S3_FORCE_PATH_STYLE", "true")

    _setup_env(mlflow_tracking_uri, s3_endpoint, aws_access_key_id, aws_secret_access_key)

    model = joblib.load(model_in.path)
    X_val = pd.read_csv(X_val_ds.path)
    y_val = pd.read_csv(y_val_ds.path).iloc[:, 0].astype(int)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_val, y_proba)),
        "f1": float(f1_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
    }

    for k, v in metrics.items():
        eval_metrics.log_metric(k, v)

    with open(auc_value, "w") as f:
        f.write(str(metrics["auc"]))

    mlflow.set_experiment("churn-kfp")
    with mlflow.start_run(run_name="evaluate_model"):
        mlflow.log_metrics(metrics)
        mlflow.log_text(json.dumps(metrics, indent=2), artifact_file="metrics.json")


@dsl.component(base_image="python:3.10-slim")
def decide_promotion(auc_value: float, threshold: float) -> str:
    msg = (
        f"AUC {auc_value:.3f} >= threshold {threshold:.3f} → ✅ ready to promote."
        if auc_value >= threshold
        else f"AUC {auc_value:.3f} < threshold {threshold:.3f} → ❌ hold promotion."
    )
    print(msg)
    return msg


@dsl.pipeline(name="churn-kfp-v2")
def churn_pipeline(
    mlflow_tracking_uri: str = "http://mlflow.kflow-mlops.svc.cluster.local:5000",
    s3_endpoint: str = "http://minio.kflow-mlops.svc.cluster.local:9000",
    aws_access_key_id: str = "admin",
    aws_secret_access_key: str = "admin123",
    auc_threshold: float = 0.70,
):
    ingest = ingest_synthetic(
        mlflow_tracking_uri=mlflow_tracking_uri,
        s3_endpoint=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    prep = preprocess_split(
        raw_dataset=ingest.outputs["output_dataset"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        s3_endpoint=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    train = train_model(
        X_train_ds=prep.outputs["X_train_ds"],
        y_train_ds=prep.outputs["y_train_ds"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        s3_endpoint=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    eval_task = evaluate_model(
        model_in=train.outputs["model_out"],
        X_val_ds=prep.outputs["X_val_ds"],
        y_val_ds=prep.outputs["y_val_ds"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        s3_endpoint=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    _ = decide_promotion(
        auc_value=eval_task.outputs["auc_value"],
        threshold=auc_threshold,
    )


# --------------------
# CLI helpers
# --------------------

def _compile(path: str = "churn_pipeline_v1.json"):
    compiler.Compiler().compile(pipeline_func=churn_pipeline, package_path=path)
    print(f"Compiled pipeline → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile the pipeline to JSON")
    args = parser.parse_args()
    if args.compile:
        _compile(); sys.exit(0)
    _compile()

