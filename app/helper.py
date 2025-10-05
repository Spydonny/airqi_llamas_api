import logging
import pandas as pd
from typing import Dict, Any
from tensorflow.keras.models import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NASA_AQI")

def calc_aqi(C, breakpoints):
    for bp in breakpoints:
        if bp["C_lo"] <= C <= bp["C_hi"]:
            return ((bp["I_hi"] - bp["I_lo"]) / (bp["C_hi"] - bp["C_lo"])) * (C - bp["C_lo"]) + bp["I_lo"]
    return None

# === Пороговые значения AQI ===
BREAKPOINTS = {
    "pm2_5": [
        {"C_lo": 0.0, "C_hi": 12.0, "I_lo": 0, "I_hi": 50},
        {"C_lo": 12.1, "C_hi": 35.4, "I_lo": 51, "I_hi": 100},
        {"C_lo": 35.5, "C_hi": 55.4, "I_lo": 101, "I_hi": 150},
        {"C_lo": 55.5, "C_hi": 150.4, "I_lo": 151, "I_hi": 200},
        {"C_lo": 150.5, "C_hi": 250.4, "I_lo": 201, "I_hi": 300},
        {"C_lo": 250.5, "C_hi": 350.4, "I_lo": 301, "I_hi": 400},
        {"C_lo": 350.5, "C_hi": 500.4, "I_lo": 401, "I_hi": 500},
    ],
    "pm10": [
        {"C_lo": 0, "C_hi": 54, "I_lo": 0, "I_hi": 50},
        {"C_lo": 55, "C_hi": 154, "I_lo": 51, "I_hi": 100},
        {"C_lo": 155, "C_hi": 254, "I_lo": 101, "I_hi": 150},
        {"C_lo": 255, "C_hi": 354, "I_lo": 151, "I_hi": 200},
        {"C_lo": 355, "C_hi": 424, "I_lo": 201, "I_hi": 300},
        {"C_lo": 425, "C_hi": 504, "I_lo": 301, "I_hi": 400},
        {"C_lo": 505, "C_hi": 604, "I_lo": 401, "I_hi": 500},
    ],
    "carbon_monoxide": [
        {"C_lo": 0.0, "C_hi": 4.4, "I_lo": 0, "I_hi": 50},
        {"C_lo": 4.5, "C_hi": 9.4, "I_lo": 51, "I_hi": 100},
        {"C_lo": 9.5, "C_hi": 12.4, "I_lo": 101, "I_hi": 150},
        {"C_lo": 12.5, "C_hi": 15.4, "I_lo": 151, "I_hi": 200},
        {"C_lo": 15.5, "C_hi": 30.4, "I_lo": 201, "I_hi": 300},
        {"C_lo": 30.5, "C_hi": 40.4, "I_lo": 301, "I_hi": 400},
        {"C_lo": 40.5, "C_hi": 50.4, "I_lo": 401, "I_hi": 500},
    ],
    "nitrogen_dioxide": [
        {"C_lo": 0, "C_hi": 53, "I_lo": 0, "I_hi": 50},
        {"C_lo": 54, "C_hi": 100, "I_lo": 51, "I_hi": 100},
        {"C_lo": 101, "C_hi": 360, "I_lo": 101, "I_hi": 150},
        {"C_lo": 361, "C_hi": 649, "I_lo": 151, "I_hi": 200},
        {"C_lo": 650, "C_hi": 1249, "I_lo": 201, "I_hi": 300},
        {"C_lo": 1250, "C_hi": 1649, "I_lo": 301, "I_hi": 400},
        {"C_lo": 1650, "C_hi": 2049, "I_lo": 401, "I_hi": 500},
    ],
    "sulphur_dioxide": [
        {"C_lo": 0, "C_hi": 35, "I_lo": 0, "I_hi": 50},
        {"C_lo": 36, "C_hi": 75, "I_lo": 51, "I_hi": 100},
        {"C_lo": 76, "C_hi": 185, "I_lo": 101, "I_hi": 150},
        {"C_lo": 186, "C_hi": 304, "I_lo": 151, "I_hi": 200},
        {"C_lo": 305, "C_hi": 604, "I_lo": 201, "I_hi": 300},
        {"C_lo": 605, "C_hi": 804, "I_lo": 301, "I_hi": 400},
        {"C_lo": 805, "C_hi": 1004, "I_lo": 401, "I_hi": 500},
    ],
    "ozone": [
        {"C_lo": 0.125, "C_hi": 0.164, "I_lo": 101, "I_hi": 150},
        {"C_lo": 0.165, "C_hi": 0.204, "I_lo": 151, "I_hi": 200},
        {"C_lo": 0.205, "C_hi": 0.404, "I_lo": 201, "I_hi": 300},
        {"C_lo": 0.405, "C_hi": 0.504, "I_lo": 301, "I_hi": 400},
        {"C_lo": 0.505, "C_hi": 0.604, "I_lo": 401, "I_hi": 500},
    ],
}
import joblib
import numpy as np

# === Параметры должны совпадать с обучением ===
LOOKBACK = 24  # окно в часах (как в обучении)
FEATURE_COLS = [
    "pm2p5_Measurement",
    "pm10_Measurement",
    "so2_Measurement",
    "o3_Measurement",
    "no2_Measurement",
    "AQI",
    "station_id_encoded",
]

TARGETS = [
    "pm2p5_Measurement_next_hour",
    "pm10_Measurement_next_hour",
    "so2_Measurement_next_hour",
    "o3_Measurement_next_hour",
    "no2_Measurement_next_hour",
    "AQI_next_hour",
]

# === Загрузка LSTM и скейлеров ===
# Папку/имена файлов поменяй под себя
LSTM_PATH = "models/pollution_lstm_model.h5"   # или .keras
SCALER_X_PATH = "models/scaler_X.pkl"
SCALER_Y_PATH = "models/scaler_y.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"  # если использовал

try:
    lstm_model = load_model(LSTM_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_Y = joblib.load(SCALER_Y_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    log.info("✅ LSTM-модель и скейлеры загружены")
except Exception as e:
    log.exception(f"❌ Не удалось загрузить LSTM/скейлеры: {e}")
    lstm_model, scaler_X, scaler_Y, le = None, None, None, None


# === AQI (US EPA) — как рассчитывали при обучении ===
def calc_us_aqi(pm25, pm10, so2, o3, no2):
    # ВНИМАНИЕ по единицам: используй те же единицы, что и в обучении!
    # (PM — μg/m3; O3/NO2/SO2 — те же, что в трейне)
    bps = {
        'pm25': [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300)],
        'pm10': [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300)],
        'so2':  [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200)],
        'o3':   [(0,0.054,0,50),(0.055,0.070,51,100),(0.071,0.085,101,150),(0.086,0.105,151,200)],
        'no2':  [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200)]
    }
    def ind(name, val):
        if val is None or np.isnan(val):
            return np.nan
        for Cl, Ch, Il, Ih in bps[name]:
            if Cl <= val <= Ch:
                return ((Ih - Il) / (Ch - Cl)) * (val - Cl) + Il
        return np.nan
    vals = [
        ind('pm25', pm25),
        ind('pm10', pm10),
        ind('so2', so2),
        ind('o3',  o3),
        ind('no2', no2),
    ]
    return np.nanmax(vals)

def build_df_from_payload(payload, station_id: str) -> pd.DataFrame:
    """
    Собираем DataFrame из AQIDataHourly (твоя входящая структура) для одной станции,
    рассчитываем AQI и код станции.
    Ожидается, что списки синхронные и упорядочены по времени (часовая частота).
    """
    df = pd.DataFrame({
        "pm10": payload.pm10,
        "pm2_5": payload.pm2_5,
        "co": payload.co,
        "no2": payload.no2,
        "so2": payload.so2,
        "o3": payload.o3,
    })
    # Если у тебя есть timestamps — подставь; иначе сделаем фиктивный RangeIndex
    # df["Timestamp_UTC"] = pd.to_datetime(payload.timestamps)  # если добавишь в схему
    # df = df.sort_values("Timestamp_UTC")

    # Переименуем в имена, с которыми обучалась модель
    df = df.rename(columns={
        "pm2_5": "pm2p5_Measurement",
        "pm10": "pm10_Measurement",
        "so2": "so2_Measurement",
        "o3":  "o3_Measurement",
        "no2": "no2_Measurement",
    })

    # AQI по строкам
    df["AQI"] = [
        calc_us_aqi(r["pm2p5_Measurement"], r["pm10_Measurement"], r["so2_Measurement"], r["o3_Measurement"], r["no2_Measurement"])
        for _, r in df.iterrows()
    ]

    # station_id_encoded
    station_code = le.transform([station_id])[0] if le else 0
    df["station_id_encoded"] = station_code

    # лёгкая интерполяция пропусков
    for c in ["pm2p5_Measurement","pm10_Measurement","so2_Measurement","o3_Measurement","no2_Measurement","AQI"]:
        df[c] = pd.Series(df[c]).interpolate(limit_direction="both")

    return df

def make_lstm_next_hour_forecast(df_last_hours: pd.DataFrame) -> Dict[str, Any]:
    """
    df_last_hours — последние LOOKBACK строк в точном порядке фичей.
    Возвращает dict с предсказаниями на следующий час по всем TARGETS.
    """
    assert lstm_model is not None, "LSTM model is not loaded"
    # берём последний срез длиной LOOKBACK
    if len(df_last_hours) < LOOKBACK:
        raise ValueError(f"Not enough history: need {LOOKBACK}, got {len(df_last_hours)}")

    window = df_last_hours.tail(LOOKBACK)[FEATURE_COLS].values  # shape (LOOKBACK, n_features)
    X = window.reshape(1, LOOKBACK, len(FEATURE_COLS))

    # масштабируем точно тем же scaler_X (внимание к reshape!)
    X_scaled = scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

    # предикт
    y_scaled = lstm_model.predict(X_scaled, verbose=0)
    # иногда сеть чуть выходит за [0,1]
    y_scaled = np.clip(y_scaled, 0.0, 1.0)

    y_inv = scaler_Y.inverse_transform(y_scaled)[0]  # shape (6,)
    result = {TARGETS[i]: float(y_inv[i]) for i in range(len(TARGETS))}
    return result

def make_lstm_multi_hour_forecast(df_last_hours: pd.DataFrame, n_hours: int) -> list[dict]:
    """
    Итеративный прогноз на n_hours вперёд.
    df_last_hours: последние LOOKBACK строк с колонками FEATURE_COLS.
    Возвращает список из n_hours словарей с ключами TARGETS.
    """
    if lstm_model is None or scaler_X is None or scaler_Y is None:
        raise RuntimeError("Model/scalers not loaded")
    if len(df_last_hours) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} rows in history")


    work = df_last_hours.tail(LOOKBACK)[FEATURE_COLS].copy().reset_index(drop=True)

    forecasts = []

    st_code = work["station_id_encoded"].iloc[-1]

    for _ in range(n_hours):
        window = work.values.reshape(1, LOOKBACK, len(FEATURE_COLS))
        X_scaled = scaler_X.transform(window.reshape(-1, window.shape[2])).reshape(window.shape)

        y_scaled = lstm_model.predict(X_scaled, verbose=0)
        y_scaled = np.clip(y_scaled, 0.0, 1.0)
        y_inv = scaler_Y.inverse_transform(y_scaled)[0]

    
        next_feature_row = {
            "pm2p5_Measurement": y_inv[TARGETS.index("pm2p5_Measurement_next_hour")],
            "pm10_Measurement":  y_inv[TARGETS.index("pm10_Measurement_next_hour")],
            "so2_Measurement":   y_inv[TARGETS.index("so2_Measurement_next_hour")],
            "o3_Measurement":    y_inv[TARGETS.index("o3_Measurement_next_hour")],
            "no2_Measurement":   y_inv[TARGETS.index("no2_Measurement_next_hour")],
            # AQI — можно взять из модели (как обучали) ИЛИ пересчитать из предсказанных поллютантов
            "AQI":               y_inv[TARGETS.index("AQI_next_hour")],
            "station_id_encoded": st_code,
        }

        step_pred = {
            "pm2p5_next_hour": next_feature_row["pm2p5_Measurement"],
            "pm10_next_hour":  next_feature_row["pm10_Measurement"],
            "so2_next_hour":   next_feature_row["so2_Measurement"],
            "o3_next_hour":    next_feature_row["o3_Measurement"],
            "no2_next_hour":   next_feature_row["no2_Measurement"],
            "AQI_next_hour":   next_feature_row["AQI"],
        }
        forecasts.append(step_pred)

        work = pd.concat([work.iloc[1:], pd.DataFrame([next_feature_row])], ignore_index=True)

    return forecasts
