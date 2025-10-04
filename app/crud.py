import xarray as xr
from shapely.geometry import Point, Polygon
from app.schemas import AQIData, AQIResponse, AQIDataHourly
from fastapi.encoders import jsonable_encoder
import httpx
from datetime import datetime, timedelta
import numpy as np
import os


from app.helper import *

client = httpx.Client(
    base_url="https://air-quality-api.open-meteo.com/v1",
    timeout=10.0,
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NASA_AQI")

# === Загрузка токена ===
load_dotenv()
NASA_TOKEN = os.getenv("NASA_TOKEN")
if not NASA_TOKEN:
    log.error("❌ NASA_TOKEN не найден в .env")
HEADERS = {"Authorization": f"Bearer {NASA_TOKEN}"}


def fetch_latest_nc_link(short_name: str) -> str:
    """Запрашивает последний .nc файл для данного short_name"""
    try:
        url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        params = {"short_name": short_name, "page_size": 1, "sort_key": "-start_date"}
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()

        data = res.json()
        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            log.warning(f"🚫 Нет записей для {short_name}")
            return None

        links = entries[0].get("links", [])
        nc = next((l["href"] for l in links if l["href"].endswith(".nc")), None)

        if nc:
            log.info(f"✅ Найден .nc файл для {short_name}: {nc}")
        else:
            log.warning(f"⚠️ Для {short_name} нет ссылок на .nc")
        return nc

    except Exception as e:
        log.exception(f"❌ Ошибка при запросе {short_name}: {e}")
        return None


def open_tempo_file(nc_url: str) -> xr.Dataset:
    """Скачивает и открывает NetCDF файл с проверками"""
    if not nc_url:
        return None

    filename = os.path.basename(nc_url)
    if not os.path.exists(filename):
        log.info(f"⬇️ Скачиваю {filename}")
        try:
            res = requests.get(nc_url, headers=HEADERS)
            res.raise_for_status()
            with open(filename, "wb") as f:
                f.write(res.content)
            log.info(f"✅ Файл сохранён: {filename}")
        except Exception as e:
            log.exception(f"❌ Ошибка при скачивании {filename}: {e}")
            return None

    try:
        ds = xr.open_dataset(filename, engine="netcdf4")
        log.info(f"📂 Файл успешно открыт: {filename}")
        return ds
    except Exception as e:
        log.exception(f"⚠️ Ошибка открытия {filename}: {e}")
        return None


def find_concentration_var(ds: xr.Dataset, pollutant: str) -> str:
    """Ищет подходящую переменную концентрации"""
    candidates = [
        v for v in ds.data_vars
        if any(k in v.lower() for k in [pollutant, "column", "conc", "vmr", "amount"])
    ]
    if candidates:
        log.info(f"🔍 Для {pollutant.upper()} найдено поле: {candidates[0]}")
        return candidates[0]
    else:
        log.warning(f"⚠️ Не найдено подходящее поле концентрации для {pollutant}")
        return None


def fetch_global_aqi() -> List[dict]:
    datasets = {
        "no2": "TEMPO_NO2_L2_NRT",
        "so2": "TEMPO_SO2_L2_NRT",
        "co": "TEMPO_CO_L2_NRT",
        "pm2_5": "TEMPO_PM25_L2_NRT",
        "pm10": "TEMPO_PM10_L2_NRT",
    }

    data = {}
    coords = None

    for pol, short_name in datasets.items():
        log.info(f"\n🌍 Обрабатываю {pol.upper()} ...")
        nc_link = fetch_latest_nc_link(short_name)
        if not nc_link:
            continue

        ds = open_tempo_file(nc_link)
        if ds is None:
            continue

        var = find_concentration_var(ds, pol)
        if not var:
            ds.close()
            continue

        try:
            # Берём первый временной слой, если есть
            if "time" in ds[var].dims:
                arr = ds[var].isel(time=0).values
                log.info(f"🕐 Выбран первый временной слой для {pol.upper()}")
            else:
                arr = ds[var].values
        except Exception as e:
            log.exception(f"❌ Ошибка извлечения данных {pol}: {e}")
            ds.close()
            continue

        lat_name = next((n for n in ["lat", "latitude", "Latitude"] if n in ds), None)
        lon_name = next((n for n in ["lon", "longitude", "Longitude"] if n in ds), None)

        if lat_name and lon_name:
            coords = (ds[lat_name].values, ds[lon_name].values)
            log.info(f"🗺️ Координаты найдены ({lat_name}, {lon_name})")

        log.info(f"✅ {pol.upper()} данные получены: shape={arr.shape}")
        data[pol] = arr
        ds.close()

    if not data:
        log.error("❌ Не удалось загрузить данные ни для одного загрязнителя")
        raise ValueError("Нет данных для AQI")

    # Приведение размеров
    shapes = [v.shape for v in data.values()]
    min_shape = tuple(np.min(shapes, axis=0))
    if len(set(shapes)) > 1:
        log.warning(f"⚠️ Разные размеры сеток {shapes}, обрезаем до {min_shape}")

    for k in data:
        if data[k].shape != min_shape:
            data[k] = data[k][:min_shape[0], :min_shape[1]]

    # Здесь можно добавить расчёт AQI
    log.info("✅ Все данные успешно собраны, готовим результат...")

    lats, lons = coords if coords else (np.arange(min_shape[0]), np.arange(min_shape[1]))
    results = []

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            results.append({
                "lat": float(lat),
                "lon": float(lon),
                **{p: float(data[p][i, j]) if p in data else None for p in data.keys()}
            })

    log.info(f"🏁 Готово! Всего точек: {len(results)}")
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

