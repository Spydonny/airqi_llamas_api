from typing import List, Optional
from pydantic import BaseModel

class AQIData(BaseModel):
    latitude: float
    longitude: float
    aqi: float
    status: str
    pm10: float = None
    pm2_5: float = None
    co: float = None
    no2: float = None
    so2: float = None

class AQIResponse(BaseModel):
    data: List[AQIData]

class AQIDataHourly(BaseModel):
    latitude: float
    longitude: float
    aqi: float
    status: str
    aqi_hourly: List[float] = None
    pm10: List[float] = None
    pm2_5: List[float] = None
    co: List[float] = None
    no2: List[float] = None
    so2: List[float] = None
    o3: List[float] = None

class PredictResponse(BaseModel):
    pm2p5_next_hour: float
    pm10_next_hour: float
    so2_next_hour: float
    o3_next_hour: float
    no2_next_hour: float
    AQI_next_hour: float

class MultiHourForecast(BaseModel):
    horizon_hours: int
    forecasts: list[dict]
