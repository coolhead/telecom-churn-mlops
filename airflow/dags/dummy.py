from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def dummy_task():
    print("Hello from dummy DAG")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 6),
}

with DAG(
    dag_id="dummy_pipeline",
    default_args=default_args,
    schedule="@daily",  # changed from schedule_interval
    catchup=False,
    tags=["dummy", "test"]
) as dag:
    t1 = PythonOperator(
        task_id="dummy_task",
        python_callable=dummy_task
    )

