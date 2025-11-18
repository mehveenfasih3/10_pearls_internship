import sys, os

import requests
import pandas as pd
import time
from datetime import datetime, timedelta


LAT=24.8607
LON=67.0011

API_KEY = os.getenv("OPENWEATHER_API_KEY")


os.makedirs("data", exist_ok=True)

def fetch_data(start, end):
    """Fetch air pollution data"""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start}&end={end}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f" Error {response.status_code} fetching data for {datetime.utcfromtimestamp(start)} → {datetime.utcfromtimestamp(end)}")
        return pd.DataFrame()

    data = response.json()
    records = []
    for item in data.get("list", []):
        dt = datetime.utcfromtimestamp(item["dt"])
        records.append({
            "datetime_utc": dt,
            "aqi": item["main"]["aqi"],
            **item["components"]
        })
    return pd.DataFrame(records)


end_time = int(time.time())  # now
part3_end = int((datetime.now() - timedelta(days=180)).timestamp())
part2_end = int((datetime.now() - timedelta(days=365)).timestamp())
part1_end = int((datetime.now() - timedelta(days=545)).timestamp())
start_time = int((datetime.now() - timedelta(days=730)).timestamp())

print(" Fetching 2 years of historical air pollution data (3-hour interval)...")

df1 = fetch_data(start_time, part1_end)
df2 = fetch_data(part1_end, part2_end)
df3 = fetch_data(part2_end, part3_end)
df4 = fetch_data(part3_end, end_time)


df = pd.concat([df1, df2, df3, df4], ignore_index=True)
df.drop_duplicates(subset="datetime_utc", inplace=True)
df.sort_values("datetime_utc", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Total records before filtering: {len(df)}")


df = df[df.index % 2 == 0]  # keeps every 3rd record ≈ every 3 hours

print(f"Total records after 3-hour interval filtering: {len(df)}")


file_path = "data/karachi_2_years.csv"
df.to_csv(file_path, index=False)
print(f" 2-year (3-hour interval) data saved to: {file_path}")
