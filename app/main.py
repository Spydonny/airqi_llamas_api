from fastapi import FastAPI
from app.crud import *
from app.schemas import AQIResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NASA_TOKEN = os.getenv("NASA_TOKEN")
headers = {"Authorization": f"Bearer {NASA_TOKEN}"}

@app.get("/air-quality/all", response_model=AQIResponse)
def get_air_quality():
    return fetch_global_aqi()
    
@app.get('/air-quality/kazakhstan', response_model=AQIResponse)
def get_air_quality_kazakhstan(step: float = 1):
    return fetch_kazakhstan_air_quality()
    
@app.get("/air-quality", response_model=AQIDataHourly)
def get_air_quality_single(latitude: float = 20.0, longitude: float = 10.0):
    return fetch_air_quality(latitude, longitude)

@app.get("/health-impact")
def get_health_recommendation(aqi: float, pm10: float, pm2_5: float, no2: float, so2: float, o3: float):
    return predict_health_impact(aqi, pm10, pm2_5, no2, so2, o3)

@app.get("/air/no2")
def get_no2():
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    params = {"short_name": "TEMPO_NO2_L2_NRT", "page_size": 1, "sort_key": "-start_date"}
    res = requests.get(url, headers=headers, params=params)
    return res.json()