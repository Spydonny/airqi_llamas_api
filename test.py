import xarray as xr
import numpy as np
import urllib.request
import os

url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v1/das/Y2025/M10/D03/GEOS-CF.v01.rpl.aqc_tavg_1hr_g1440x721_v1.20251003_0030z.nc4"
filename = os.path.basename(url)

# 1️⃣ Скачать файл локально, если его ещё нет
if not os.path.exists(filename):
    print(f"Скачиваю {filename} ...")
    urllib.request.urlretrieve(url, filename)

# 2️⃣ Открыть локальный файл через netCDF4
ds = xr.open_dataset(filename, engine="netcdf4")

print(ds.variables)  # Показать все переменные в наборе данных