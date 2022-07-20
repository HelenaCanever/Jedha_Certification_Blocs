import os
import argparse
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import  StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor


if __name__ == "__main__":

    # Set your variables for your environment
    EXPERIMENT_NAME="test-experiment"

    # Instanciate your experiment
    client = mlflow.tracking.MlflowClient()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Set experiment's info 
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    run = client.create_run(experiment.experiment_id) # Creates a new run for a given experiment
    
    print("training model...")
    
    mlflow.xgboost.autolog()

    # Import dataset
    df = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv", index_col=[0])

    #remove outliers in mileage and engine_power columns
    mileage_filter = ((df['mileage'].mean() - 3*df['mileage'].std()) < df['mileage']) & (df['mileage'] < (df['mileage'].mean() + 3*df['mileage'].std())) 
    engine_filter = ((df['engine_power'].mean() - 3*df['engine_power'].std()) < df['engine_power']) & (df['engine_power'] < (df['engine_power'].mean() + 3*df['engine_power'].std())) 
    filters = mileage_filter & engine_filter
    df_clean = df.loc[filters,:]

    # divide target and features
    Y = df_clean.loc[:,'rental_price_per_day']
    X = df_clean.iloc[:,0:13]

    # Define feature processing
    numeric_features = ["mileage", "engine_power"]
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
        ])

    categorical_features = ["model_key", "fuel", "paint_color", "car_type", "private_parking_available", "has_gps", "has_air_conditioning", "automatic_car", "has_getaround_connect", "has_speed_regulator", "winter_tires"]
    categorical_transformer = Pipeline(
        steps=[
        ('encoder', OneHotEncoder(drop='first'))
        ])

    # Process X
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
    ("Preprocessing", preprocessor),
    ("Model",XGBRegressor(max_depth=4))
    ]) 

    # Log experiment to MLFlow
    with mlflow.start_run(run_id = run.info.run_id) as run:
        #debug
        print(mlflow.get_artifact_uri())
        print(mlflow.get_tracking_uri())

        model.fit(X, Y)
        predictions = model.predict(X)
        print(predictions)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="getaround_estimator",
            registered_model_name="xgbmodel",
            signature=infer_signature(X, predictions)
        )
        
    print("...Done!")