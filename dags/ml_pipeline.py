from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from scripts.model_promoter import model_promoter
from scripts.log_and_save import log_and_save
from scripts.preprocess_data import preprocess_data
from scripts.pull_data_from_dvc import pull_data_from_dvc
from scripts.train_model import train_model

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'clothes_classification_training_pipeline',
    default_args=default_args,
    description='DAG for Clothes Classification Training Pipeline',
    schedule_interval=timedelta(days=1),
)

def pull_data_from_dvc_wrapper(**kwargs):
    pull_data_from_dvc()

def preprocess_data_wrapper(**kwargs):
    preprocess_data(data_dir=kwargs['data_dir'])

def train_model_wrapper(**kwargs):
    train_model(data_dir=kwargs['data_dir'])

def model_promoter_wrapper(**kwargs):
    model_promoter()

def log_and_save_wrapper(**kwargs):
    log_and_save(
        model_dir=kwargs['model_dir'],
        promotion_decision=kwargs['promotion_decision'],
        model_path=kwargs['model_path'],
        dvc_file_path=kwargs['dvc_file_path'],
        registered_model_name=kwargs['registered_model_name']
    )

# Define the tasks
pull_data_from_dvc_task = PythonOperator(
    task_id='pull_data_from_dvc',
    python_callable=pull_data_from_dvc_wrapper,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_wrapper,
    op_kwargs={'data_dir': 'data'},
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_wrapper,
    op_kwargs={'data_dir': 'data'},
    dag=dag,
)

model_promoter_task = PythonOperator(
    task_id='model_promoter',
    python_callable=model_promoter_wrapper,
    dag=dag,
)

log_and_save_task = PythonOperator(
    task_id='log_and_save',
    python_callable=log_and_save_wrapper,
    op_kwargs={
        'model_dir': 'models/model.h5',
        'promotion_decision': True,
        'model_path': 'models',
        'dvc_file_path': 'data.dvc',
        'registered_model_name': 'mobile_net'
    },
    dag=dag,
)

# Define task dependencies
pull_data_from_dvc_task >> preprocess_data_task >> train_model_task >> model_promoter_task >> log_and_save_task
