import requests, os
from dotenv import load_dotenv

load_dotenv()
NASA_TOKEN = os.getenv("NASA_TOKEN")
headers = {"Authorization": f"Bearer {NASA_TOKEN}"}

url = "https://cmr.earthdata.nasa.gov/search/collections.json"
params = {"keyword": "TEMPO", "page_size": 100}
res = requests.get(url, headers=headers, params=params)
data = res.json()

print(f"{'SHORT_NAME':35} | {'VERSION':7} | {'DATA CENTER':15} | {'SUMMARY'}")
print("-" * 100)

for entry in data["feed"]["entry"]:
    short_name = entry.get("short_name", "")
    version = entry.get("version_id", "")
    center = entry.get("data_center", "")
    summary = entry.get("summary", "").replace("\n", " ").strip()
    if len(summary) > 80:  # ограничим вывод
        summary = summary[:77] + "..."
    print(f"{short_name:35} | {version:7} | {center:15} | {summary}")
