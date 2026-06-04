import pandas as pd
import pickle
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import yaml
import os
import mlflow
from urllib.parse import urlparse


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/brajrajputofficial/ml_pipeline_with_mlflow_projecy.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "brajrajputofficial"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "f01d1e3006e34deb71d36df5aaf8a49613c143a7"

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]


    # mlflow.set_tracking_uri("https://dagshub.com/brajrajputofficial/ml_pipeline_with_mlflow_projecy.mlflow")

    ## load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)

if __name__=="__main__":
    evaluate(params["data"],params["model"])
