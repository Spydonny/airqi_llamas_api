import xarray as xr
from shapely.geometry import Point, Polygon
from app.schemas import AQIData, AQIResponse, AQIDataHourly
import httpx
from datetime import datetime, timedelta
import fsspec
import numpy as np
import urllib.request
import os


from app.helper import *

client = httpx.Client(
    base_url="https://air-quality-api.open-meteo.com/v1",
    timeout=10.0,
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
)

def get_aqi_status(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def frange(start: float, stop: float, step: float):
    while start <= stop:
        yield round(start, 5)
        start += step

# Упрощённый полигон Казахстана (для примера; лучше подставить полный из GeoJSON)
from shapely.geometry import Polygon, Point

# Полигон Бельгии (приближённый контур)
BELGIUM_POLYGON = Polygon([
    (2.51357303225, 51.1485061713),   # северо-запад (граница с Францией и Северным морем)
    (3.31497114423, 51.3457809515),
    (4.04707116051, 51.2672586127),
    (4.97399132662, 51.4750237087),
    (5.60697594567, 51.03729848897),  # северо-восток (граница с Нидерландами/Германией)
    (6.15665815596, 50.803721015),
    (6.04307335778, 50.1280516628),
    (5.7824174333, 50.0903278672),
    (5.67405195478, 49.5294835476),   # юго-восток (граница с Люксембургом)
    (4.79922163252, 49.9853730332),
    (4.28602298343, 49.9074966498),
    (3.58818444176, 50.3789924180),
    (2.56870071544, 50.4020917622),
    (2.51357303225, 51.1485061713)    # замыкание полигона
])

def fetch_kazakhstan_air_quality(step: float = 0.5) -> AQIResponse:
    """
    Получаем качество воздуха внутри полигона Бельгии.
    step - шаг сетки (в градусах).
    """
    lat_start, lat_end = 49.5, 51.5
    lon_start, lon_end = 2.5, 6.5

    result = []
    for lat in frange(lat_start, lat_end, step):
        for lon in frange(lon_start, lon_end, step):
            point = Point(lon, lat)  # shapely: (x=lon, y=lat)
            if not BELGIUM_POLYGON.contains(point):
                continue

            params = {
                "latitude": lat,
                "longitude": lon,
                "current": [
                    "european_aqi",
                    "sulphur_dioxide",
                    "pm10",
                    "pm2_5",
                    "carbon_monoxide",
                    "nitrogen_dioxide",
                ],
            }
            r = client.get("/air-quality", params=params)
            if r.status_code != 200:
                continue

            data = r.json()
            if "current" in data:
                current_aqi = data["current"]["european_aqi"]
                result.append(AQIData(
                    latitude=data["latitude"],
                    longitude=data["longitude"],
                    aqi=current_aqi,
                    pm10=data["current"].get("pm10"),
                    pm2_5=data["current"].get("pm2_5"),
                    co=data["current"].get("carbon_monoxide"),
                    no2=data["current"].get("nitrogen_dioxide"),
                    so2=data["current"].get("sulphur_dioxide"),
                    status=get_aqi_status(current_aqi),
                ))

    return AQIResponse(data=result)


def fetch_global_aqi() -> list[AQIData]:
    """
    Загружает глобальные данные качества воздуха из NetCDF-источника (GEOS-CF)
    и рассчитывает AQI по всей сетке.
    """

    url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v1/das/Y2025/M10/D03/GEOS-CF.v01.rpl.aqc_tavg_1hr_g1440x721_v1.20251003_0030z.nc4"
    filename = os.path.basename(url)

    # 1️⃣ Скачать файл локально, если его ещё нет
    if not os.path.exists(filename):
        print(f"Скачиваю {filename} ...")
        urllib.request.urlretrieve(url, filename)

    # 2️⃣ Открыть файл через xarray
    ds = xr.open_dataset(filename, engine="netcdf4")

    # 3️⃣ Проверяем наличие доступных переменных
    var_map = {
        "pm2_5": ["PM25_RH35_GCC", "pm2p5_conc", "PM25", "PM2_5"],
        "pm10": ["PM10", "pm10_conc"],
        "no2":  ["NO2", "no2_conc"],
        "so2":  ["SO2", "so2_conc"],
        "co":   ["CO", "co_conc"]
    }

    resolved_vars = {}

    for key, options in var_map.items():
        for v in options:
            if v in ds.data_vars:
                resolved_vars[key] = v
                break
        else:
            print(f"⚠️ Переменная для {key} не найдена в наборе данных")

    # Проверим, что хотя бы PM2.5 есть
    if "pm2_5" not in resolved_vars:
        raise ValueError("PM2.5 data not found in dataset — cannot compute AQI")

    # 4️⃣ Извлекаем данны

    data = {}
    for pollutant, var_name in resolved_vars.items():
        data[pollutant] = ds[var_name].isel(time=0, lev=0).values

    lats = ds["lat"].values
    lons = ds["lon"].values

    # 5️⃣ Рассчитываем AQI по каждому загрязнителю
    pollutant_aqi = {}
    for pol, arr in data.items():
        # Для каждого загрязнителя используем свои пороги
        if pol in BREAKPOINTS:
            pollutant_aqi[pol] = np.vectorize(lambda c: calc_aqi(c, BREAKPOINTS[pol]))(arr)
        else:
            pollutant_aqi[pol] = np.full_like(arr, np.nan)

    # 6️⃣ Общий AQI — максимум среди всех загрязнителей
    stacked = np.stack(list(pollutant_aqi.values()), axis=0)
    overall_aqi = np.nanmax(stacked, axis=0)

    # 7️⃣ Конвертируем в список AQIData
    results = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            aqi_val = overall_aqi[i, j]
            if np.isnan(aqi_val):
                continue  # пропускаем точки без данных

            # Безопасное преобразование с NaN → None
            def safe_float(x):
                return float(x) if x is not None and not np.isnan(x) else None

            results.append(
                AQIData(
                    latitude=float(lat),
                    longitude=float(lon),
                    aqi=safe_float(aqi_val),
                    status=get_aqi_status(aqi_val),
                    pm10=safe_float(data["pm10"][i, j]) if "pm10" in data else None,
                    pm2_5=safe_float(data["pm2_5"][i, j]),
                    co=safe_float(data["carbon_monoxide"][i, j]) if "carbon_monoxide" in data else None,
                    no2=safe_float(data["nitrogen_dioxide"][i, j]) if "nitrogen_dioxide" in data else None,
                    so2=safe_float(data["sulphur_dioxide"][i, j]) if "sulphur_dioxide" in data else None,
                )
            )
    ds.close()
    return results


def fetch_air_quality(latitude, longitude) -> AQIDataHourly:
    today = datetime.today().date()
    one_month_before = today - timedelta(days=14)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "european_aqi",
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
        "start_date": one_month_before.strftime("%Y-%m-%d"),
        "end_date": today.strftime("%Y-%m-%d"),
    }

    r = client.get("/air-quality", params=params)
    r.raise_for_status()
    data = r.json()

    current_aqi = data["current"]["european_aqi"]

    # --- расчёт почасового AQI ---
    n = len(data["hourly"]["pm2_5"])
    hourly_aqi = []
    for i in range(n):
        vals = {
            "pm2_5": data["hourly"]["pm2_5"][i],
            "pm10": data["hourly"]["pm10"][i],
            "carbon_monoxide": data["hourly"]["carbon_monoxide"][i],
            "nitrogen_dioxide": data["hourly"]["nitrogen_dioxide"][i],
            "sulphur_dioxide": data["hourly"]["sulphur_dioxide"][i],
            "ozone": data["hourly"]["ozone"][i],
        }
        # расчёт AQI для каждого загрязнителя
        aqi_values = []
        for k, v in vals.items():
            aqi_val = calc_aqi(v, BREAKPOINTS[k])
            if aqi_val is not None:
                aqi_values.append(aqi_val)

        hourly_aqi.append(max(aqi_values) if aqi_values else None)

    result = AQIDataHourly(
        latitude=data["latitude"],
        longitude=data["longitude"],
        aqi=current_aqi,
        aqi_hourly=hourly_aqi,
        status=get_aqi_status(current_aqi),
        pm10=data["hourly"]["pm10"],
        pm2_5=data["hourly"]["pm2_5"],
        co=data["hourly"]["carbon_monoxide"],
        no2=data["hourly"]["nitrogen_dioxide"],
        so2=data["hourly"]["sulphur_dioxide"],
        o3=data["hourly"]["ozone"],
    )
    return result

