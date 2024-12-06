# MLops Project 

I will explain how we can use DVC, Airflow and MLflow to efficiently manage our data pipelines.
##Installation

```bash
pip install dvc dvc[gdrive] mlflow 
```
## Usage
To run data collection script:
```bash
python data_collection.py
```
To process the data to prepare it for model input:
```bash
python preprocess_data.py
```
To use ml server (open a different terminal to run this):
```bash
 mlflow server --host 127.0.0.1 --port 8080
```
To train your model and log paramaters and model on mlflow:
```bash
python train_model.py
```
To run docker image for airflow:
```bash
docker-compose up
```
To run your DAG locally, copy the dag folder path and do:
```bash
python myDAG.py
```
