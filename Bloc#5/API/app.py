import mlflow 
import uvicorn
import pandas as pd 
import numpy as np
from pydantic import BaseModel
from typing import Literal
from fastapi import FastAPI, File, UploadFile

description = """
This is a GetAround pricing prediction API

## Prediction Endpoint
To obtain a prediction of the best pricing for your car:
* `/predict`: accepts car specifications (see schema below) and returns a price prediction based on input data.

"""


tags_metadata = [
    {
        "name": "Prediction Endpoint",
    }
]

app = FastAPI(
    title="🔮 GetAround Price Predictor",
    description=description,
    openapi_tags=tags_metadata
)


class PredictionInput(BaseModel):
    model_key: Literal['Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Ford',
       'Mercedes', 'Opel', 'Porsche', 'Volkswagen', 'KIA Motors',
       'Alfa Romeo', 'Ferrari', 'Fiat', 'Lamborghini', 'Maserati',
       'Lexus', 'Honda', 'Mazda', 'Mini', 'Mitsubishi', 'Nissan', 'SEAT',
       'Subaru', 'Suzuki', 'Toyota', 'Yamaha'] = "Citroën"
    mileage: int = 141080
    engine_power: int = 120
    fuel: Literal['diesel', 'petrol', 'hybrid_petrol', 'electro'] = "diesel"
    paint_color: Literal['black', 'grey', 'white', 'red', 'silver', 'blue', 'orange',
       'beige', 'brown', 'green'] = "black"
    car_type: Literal['convertible', 'coupe', 'estate', 'hatchback', 'sedan',
       'subcompact', 'suv', 'van'] = "estate"
    private_parking_available: Literal[True, False] = True
    has_gps: Literal[True, False] = True
    has_air_conditioning: Literal[True, False] = False
    automatic_car: Literal[True, False] = True
    has_getaround_connect: Literal[True, False] = True
    has_speed_regulator: Literal[True, False] = True
    winter_tires: Literal[True, False] = False


@app.get("/")
async def index():
    message = "Welcome to the GetArond prediction API. Please check out the documentation of the api at `/docs`"
    return message


@app.post("/predict", tags=["Prediction Endpoint"])
async def predict(PredictionInput: PredictionInput):

    # Read input data 
    prediction_input = pd.DataFrame({"model_key": [PredictionInput.model_key],
    "mileage": [PredictionInput.mileage],
    "engine_power": [PredictionInput.engine_power],
    "fuel": [PredictionInput.fuel],
    "paint_color": [PredictionInput.paint_color],
    "car_type": [PredictionInput.car_type],
    "private_parking_available": [PredictionInput.private_parking_available],
    "has_gps": [PredictionInput.has_gps],
    "has_air_conditioning": [PredictionInput.has_air_conditioning],
    "automatic_car": [PredictionInput.automatic_car],
    "has_getaround_connect": [PredictionInput.has_getaround_connect],
    "has_speed_regulator": [PredictionInput.has_speed_regulator],
    "winter_tires": [PredictionInput.winter_tires]
    })

    # Log model from mlflow 
    logged_model = 'runs:/42a196c5ca6745a8960e1a8f29ac311c/getaround_estimator'

    mlflow.pyfunc.get_model_dependencies(logged_model)
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    prediction = loaded_model.predict(prediction_input)

    # Format response
    response = {"prediction": prediction.tolist()[0]}
    return response


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)