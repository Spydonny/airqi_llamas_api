from fastapi import FastAPI
from app.crud import *
from app.schemas import AQIResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/air-quality/all", response_model=AQIResponse)
def get_air_quality():
    return fetch_global_aqi()
    
@app.get('/air-quality/kazakhstan', response_model=AQIResponse)
def get_air_quality_kazakhstan(step: float = 1):
    return fetch_kazakhstan_air_quality(step)
    
@app.get("/air-quality", response_model=AQIDataHourly)
def get_air_quality_single(latitude: float = 20.0, longitude: float = 10.0):
    return fetch_air_quality(latitude, longitude)

# @app.get("/health")
# def get_health_recommendation(aqi: int):
#     return fetch_health_recommendation(aqi)

