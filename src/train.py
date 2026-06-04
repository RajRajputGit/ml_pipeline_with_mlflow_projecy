import pandas as pd
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import yaml
import pickle
from mlflow.models import infer_signature
import mlflow
from urllib.parse import urlparse


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/brajrajputofficial/ml_pipeline_with_mlflow_projecy.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "brajrajputofficial"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "f01d1e3006e34deb71d36df5aaf8a49613c143a7"

#loaad the parameters from yaml file
params = yaml.safe_load(open("params.yaml","rb"))["train"]

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# Train the model
def train_model(data_path,model_path,random_state,n_estimators,max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)

    # hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2,5],
        'min_samples_leaf':[1,2]
    }

    # finding the best model using gridsearchcv
    grid_search = hyperparameter_tuning(X_train,y_train,param_grid=param_grid)
    best_model = grid_search.best_estimator_

    #predictions
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)

    print(f"accuracy: {accuracy}")

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)


    #mlflow tracking
    mlflow.set_tracking_uri("https://dagshub.com/brajrajputofficial/ml_pipeline_with_mlflow_projecy.mlflow")

    #logging the parameters and metrics using mlflow run
    with mlflow.start_run():

        signature = infer_signature(X_train[:5], best_model.predict(X_train[:5]))

        #logging the best model param
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_estimators", best_model.n_estimators)
        mlflow.log_param("max_depth", best_model.max_depth)
        mlflow.log_param("min_samples_split", best_model.min_samples_split)
        mlflow.log_param("min_samples_leaf", best_model.min_samples_leaf)

        #logging the metrics
        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="RF model with best params")
        else:
             mlflow.sklearn.log_model(best_model, "Rf model with best param", signature=signature)

        # save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_model(params["data_path"],params["model_path"],params["random_state"],params["n_estimators"],params["max_depth"])