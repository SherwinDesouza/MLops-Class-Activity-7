from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator

from preprocess_data import processing  
from data_collection import fetch_weather_data
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 11, 27),
}

with DAG(
    'weather_pipeline',
    default_args=default_args,
    schedule_interval=None,  
    catchup=False,
) as dag:



    collect_data_task = PythonOperator(
        task_id='collect_data_task',
        python_callable=fetch_weather_data, 
    )

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data_task',
        python_callable=processing, 
    )

    DVCWalaKaam = BashOperator(
        task_id="DVCWalaKaam",
        bash_command=(
            'dvc add dags/raw_data.csv dags/processed_data.csv && '
            'dvc push'
        ),
        retries=3,
        execution_timeout=timedelta(minutes=15),
    )
    


collect_data_task >> preprocess_data_task >> DVCWalaKaam

