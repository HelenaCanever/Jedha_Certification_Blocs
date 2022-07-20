

## Bloc#5 GetAround

### Contact: helena.canever@gmail.com
### Video link:
### Deliverables:
  - Dashboard : https://getarounddashboardhc.herokuapp.com/
  - MLFlow Server: https://getaroundflow.herokuapp.com/
  - API : https://getaroundapi.herokuapp.com/

### API Predict

With Python:

```
import requests

response = requests.post("https://getaroundapi.herokuapp.com/predict", json={"model_key": "Toyota", "mileage": 141080, "engine_power": 120, "fuel": "hybrid_petrol", "paint_color": "silver", "car_type": "sedan", "private_parking_available": True, "has_gps": True, "has_air_conditioning": True, "automatic_car": True, "has_getaround_connect": True, "has_speed_regulator": True,"winter_tires": False})
print(response.json())
```

With curl:

```
$curl -X 'POST' \
  'https://getaroundapi.herokuapp.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"model_key": "Toyota", "mileage": 141080, "engine_power": 120, "fuel": "hybrid_petrol", "paint_color": "silver","car_type": "sedan","private_parking_available": true,"has_gps": true,"has_air_conditioning": true,"automatic_car": true,"has_getaround_connect": true,"has_speed_regulator": true,"winter_tires": false}'
  
 ```
