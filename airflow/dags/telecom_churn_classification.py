from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import os
import sys

# Set path to scripts
dag_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dag_dir, '..', '..', 'upgrad', 'pipeline', 'telecom-churn-mlops'))
sys.path.insert(0, os.path.join(project_root, 'scripts'))

# Import scripts after sys.path update
import telecom_preprocessing
import telecom_model_inference
import telecom_drift_check
import telecom_retrain_model
from add_mlflow_dag_airflow import register_latest_model


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 1),
    'retries': 1
}


register_model_task = PythonOperator(
    task_id='register_latest_model',
    python_callable=register_latest_model
)



with DAG(
    'telecom_churn_classification',
    default_args=default_args,
    schedule='@daily',
    catchup=False
) as dag:

    start = EmptyOperator(task_id='start')

    preprocess_task = PythonOperator(
        task_id='telecom_preprocess_data',
        python_callable=telecom_preprocessing.telecom_preprocess_data
    )

    predict_task = PythonOperator(
        task_id='predict_with_registered_model',
        python_callable=telecom_model_inference.telecom_load_model_and_predict
    )

    drift_task = PythonOperator(
        task_id='check_data_drift',
        python_callable=telecom_drift_check.telecom_monitor_drift
    )

    retrain_task = PythonOperator(
        task_id='retrain_model_if_drift',
        python_callable=telecom_retrain_model.telecom_retrain_and_log_model
    )


    start >> preprocess_task >> predict_task >> drift_task >> retrain_task >> register_model_task
