from fastapi import FastAPI
from pydantic import BaseModel
# # from typing import List
# # from fastapi.responses import FileResponse
import pandas as pd
import json
import pickle
import sklearn
from typing import List
from fastapi.responses import FileResponse
from fastapi import UploadFile
from typing import Annotated
from fastapi import File
import numpy as np
from fastapi.responses import StreamingResponse

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    seats: int

class Items(BaseModel):
    objects: List[Item]

cars_model = pickle.load(open("cars_model.pkl", "rb"))
scaler = pickle.load(open("standard_scaler.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    input_data = item.json()
    input_dictionary = json.loads(input_data)
    # name = input_dictionary["name"]
    year = input_dictionary["year"]
    # selling_price = input_dictionary["selling_price"]
    km_driven = input_dictionary["km_driven"]
    fuel = input_dictionary["fuel"]
    seller_type = input_dictionary["seller_type"]
    transmission = input_dictionary["transmission"]
    owner = input_dictionary["owner"]
    mileage = input_dictionary["mileage"]
    engine = input_dictionary["engine"]
    max_power = input_dictionary["max_power"]
    seats = input_dictionary["seats"]
    my_list = [year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]
    my_list.append(max_power/engine)
    my_list.append(year**2)
    my_list = pd.DataFrame([my_list], columns = ["year", "km_driven", "fuel", "seller_type", "transmission",
                                               "owner", "mileage", "engine", "max_power", "seats", "max_power_engine", "year_squared"])
    tramsformed_list = transformer.transform(my_list)
    scaled_list = scaler.transform(tramsformed_list)
    prediction = cars_model.predict(scaled_list)
    return prediction

@app.post("/uploadcsv/")
def upload_csv(csv_file: UploadFile = File(...)) -> List[float]:
    df = pd.read_csv(csv_file.file)
    tramsformed_list = transformer.transform(df)
    scaled_list = scaler.transform(tramsformed_list)
    df["selling_price_pred"] = cars_model.predict(scaled_list)
    return df["selling_price_pred"]




