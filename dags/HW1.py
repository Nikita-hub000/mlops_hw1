from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from airflow.utils.dates import days_ago, timedelta
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import json
import time
import pickle
DEFAULT_ARGS = {
    "owner": "Lapshin Nikita",
    "retries": 3,
    "retry_delay": timedelta(minutes=1)
}

BUCKET = Variable.get("S3_BUCKET")
model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ])
)

def init(m_name: str):
    timestamp = time.time()
    print(f"Init DAG for {m_name}")
    return {"timestamp": timestamp, "model_name": m_name}

def get_data(**kwargs):
    ti = kwargs['ti']
    model_metadata = ti.xcom_pull(task_ids='init')
    
    start_time = time.time()
    
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    
    end_time = time.time()
    
    dataset_info = {
        "start_time": start_time,
        "end_time": end_time,
        "size": len(data.data),
        "features": list(data.data.columns)
    }

    s3_hook = S3Hook("s3_connection")
    path = f"LapshinNikita/{model_metadata['model_name']}/datasets/california_housing.csv"
    s3_hook.load_string(data.data.to_csv(), path, bucket_name=BUCKET, replace=True)
    
    return dataset_info

def prepare_data(**kwargs):
    ti = kwargs['ti']
    model_metadata = ti.xcom_pull(task_ids='init')
    
    s3_hook = S3Hook("s3_connection")
    path = f"LapshinNikita/{model_metadata['model_name']}/datasets/california_housing.csv"
    data = pd.read_csv(s3_hook.download_file(path, bucket_name=BUCKET))
    
    start_time = time.time()
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, :-1])  
    y = data.iloc[:, -1] 
    
    end_time = time.time()

    prepared_data_info = {
        "start_time": start_time,
        "end_time": end_time,
        "features": list(data.columns[:-1])
    }
    
    prepared_path = f"LapshinNikita/{model_metadata['model_name']}/datasets/prepared_data.csv"
    prepared_data = pd.DataFrame(X)
    s3_hook.load_string(prepared_data.to_csv(), prepared_path, bucket_name=BUCKET, replace=True)
    
    return prepared_data_info

def train_model(**kwargs):
    ti = kwargs['ti']
    model_metadata = ti.xcom_pull(task_ids='init')
    
    s3_hook = S3Hook("s3_connection")
    prepared_path = f"LapshinNikita/{model_metadata['model_name']}/datasets/prepared_data.csv"
    data = pd.read_csv(s3_hook.download_file(prepared_path, bucket_name=BUCKET))
    
    X = data.values[:, :-1]
    y = data.values[:, -1]

    start_time = time.time()

    model = models[model_metadata['model_name']]
    model.fit(X, y)
    
    end_time = time.time()
    
    model_metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "model_score": model.score(X, y)
    }
    
    model_path = f"LapshinNikita/{model_metadata['model_name']}/results/model.pkl"
    s3_hook.load_bytes(pickle.dumps(model), key=model_path, bucket_name=BUCKET, replace=True)
    
    return model_metrics

def save_results(**kwargs):
    ti = kwargs['ti']
    model_metadata = ti.xcom_pull(task_ids='init')
    model_metrics = ti.xcom_pull(task_ids='train_model')
    
    s3_hook = S3Hook("s3_connection")
    results_path = f"LapshinNikita/{model_metadata['model_name']}/results/metrics.json"
    s3_hook.load_string(json.dumps(model_metrics), results_path, bucket_name=BUCKET, replace=True)
    
    print("Results saved successfully!")

def create_dag(dag_id: str, m_name: str):
    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(1),
        default_args=DEFAULT_ARGS,
        tags=["mlops"]
    )

    with dag:
        task_init = PythonOperator(
            task_id="init",
            python_callable=init,
            op_kwargs={"m_name": m_name}
        )

        task_get_data = PythonOperator(
            task_id="get_data",
            python_callable=get_data,
        )

        task_prepare_data = PythonOperator(
            task_id="prepare_data",
            python_callable=prepare_data,
        )

        task_train_model = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
        )

        task_save_results = PythonOperator(
            task_id="save_results",
            python_callable=save_results,
        )

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results

    return dag

# Создаем DAGи для каждой модели
for model_name in models.keys():
    globals()[f"Nikita_Lapshin_{model_name}"] = create_dag(f"Nikita_Lapshin_{model_name}", model_name)
