from typing import List
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