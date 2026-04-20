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

zones = {
    "snell_floor2_silent":    {"base": 42.0, "noise": 3.5},
    "healthcare_reflection":  {"base": 38.0, "noise": 2.5},
}

start_time = datetime.now(timezone.utc) - timedelta(hours=1, minutes=5)
interval   = 30   # seconds
n_points   = 121  # 1 hour + buffer

points = []
for i in range(n_points):
    ts = start_time + timedelta(seconds=i * interval)
    for zone, cfg in zones.items():
        # Diurnal variation + random noise + occasional spike
        diurnal = 2.0 * math.sin(2 * math.pi * i / n_points)
        spike   = random.choices([0, random.uniform(8, 15)],
                                  weights=[0.92, 0.08])[0]
        dba     = round(cfg["base"] + diurnal
                        + random.gauss(0, cfg["noise"]) + spike, 1)
        dba     = max(28.0, min(75.0, dba))
        status  = "violation" if dba >= 45.0 else "compliant"

        p = (Point("sound_level")
             .tag("zone_id",    zone)
             .tag("status",     status)
             .field("dba_spl",  dba)
             .time(ts, WritePrecision.S))
        points.append(p)

write_api.write(bucket=BUCKET, org=ORG, record=points)
print(f"Written {len(points)} points to InfluxDB ({n_points} readings x {len(zones)} zones)")
client.close()
