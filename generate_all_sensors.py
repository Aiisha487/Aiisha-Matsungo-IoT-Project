from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone, timedelta
import random, math

TOKEN  = "seminar-test-token"
ORG    = "eece5155"
BUCKET = "iot-sensors"
URL    = "http://localhost:8086"

client    = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Three Snell Library zones with different acoustic profiles
zones = {
    "snell_silent_study": {
        "base": 38.0, "noise": 4.5, "threshold": 40.0,
        "temp_base": 20.5, "hum_base": 45.0, "light_base": 180.0,
        "description": "Small carrel 10m2 - strict quiet zone"
    },
    "snell_reading_hall": {
        "base": 42.0, "noise": 3.0, "threshold": 45.0,
        "temp_base": 21.5, "hum_base": 48.0, "light_base": 350.0,
        "description": "Large open floor 200m2 - standard reading area"
    },
    "snell_group_room": {
        "base": 48.0, "noise": 5.0, "threshold": 55.0,
        "temp_base": 23.0, "hum_base": 52.0, "light_base": 420.0,
        "description": "Medium collaborative room 30m2 - relaxed noise policy"
    },
}

start_time = datetime.now(timezone.utc) - timedelta(hours=1, minutes=5)
interval   = 30   # seconds
n_points   = 121  # 1 hour of data

points = []
for i in range(n_points):
    ts = start_time + timedelta(seconds=i * interval)

    for zone, cfg in zones.items():
        # --- Sound level (MAX9814) ---
        diurnal = 2.0 * math.sin(2 * math.pi * i / n_points)
        # Small room gets sharper spikes due to reflections
        spike_chance = 0.12 if zone == "snell_silent_study" else 0.07
        spike = random.choices(
            [0, random.uniform(6, 18)],
            weights=[1 - spike_chance, spike_chance]
        )[0]
        dba = round(cfg["base"] + diurnal +
                    random.gauss(0, cfg["noise"]) + spike, 1)
        dba = max(25.0, min(78.0, dba))
        status = "violation" if dba >= cfg["threshold"] else "compliant"

        points.append(
            Point("sound_level")
            .tag("zone_id", zone)
            .tag("status",  status)
            .field("dba_spl",    dba)
            .field("threshold",  cfg["threshold"])
            .time(ts, WritePrecision.S)
        )

        # --- Temperature & Humidity (DHT22) ---
        temp = round(cfg["temp_base"] +
                     1.2 * math.sin(2 * math.pi * i / n_points) +
                     random.gauss(0, 0.3), 1)
        hum  = round(cfg["hum_base"] +
                     2.5 * math.sin(2 * math.pi * i / n_points) +
                     random.gauss(0, 1.0), 1)

        points.append(
            Point("environment")
            .tag("zone_id", zone)
            .tag("sensor",  "DHT22")
            .field("temperature_c", temp)
            .field("humidity_pct",  hum)
            .time(ts, WritePrecision.S)
        )

        # --- Light level (LDR - occupancy proxy) ---
        # Light rises mid-session (more people arrive)
        occupancy_curve = cfg["light_base"] + \
            40.0 * math.sin(math.pi * i / n_points)
        light = round(occupancy_curve + random.gauss(0, 12.0), 1)
        light = max(50.0, light)

        points.append(
            Point("environment")
            .tag("zone_id", zone)
            .tag("sensor",  "LDR")
            .field("light_lux", light)
            .time(ts, WritePrecision.S)
        )

write_api.write(bucket=BUCKET, org=ORG, record=points)
print(f"Written {len(points)} points to InfluxDB")
print(f"  - {n_points * len(zones)} sound_level readings across {len(zones)} zones")
print(f"  - {n_points * len(zones)} DHT22 readings (temp + humidity)")
print(f"  - {n_points * len(zones)} LDR readings (light/occupancy)")
client.close()
