# Acoustic Monitoring System — EECE 5155 Final Deliverable

**Student:** Aiisha Matsungo  
**NUID:** 002530298  
**Course:** EECE 5155 Wireless Sensor Networks and IoT Systems  
**Semester:** Spring 2026  
**Wokwi Project:** https://wokwi.com/projects/461101098635750401  
**GitHub:** https://github.com/Aiisha487/Aiisha-Matsungo-IoT-Project

## Project Description
An IoT acoustic monitoring system for quiet zone compliance in institutional facilities (Snell Library, Northeastern University). Three zones are modelled: silent study carrel (40 dBA threshold), reading hall (45 dBA), and group room (55 dBA). The system publishes dBA SPL readings via MQTT, stores data in InfluxDB, visualises in Grafana, and applies an SVM classifier trained on the UCI HAR dataset to demonstrate the full MLOps pipeline.

## Repository Structure
## Quick Start

### Pipeline (Part A)
1. `git clone https://github.com/Aiisha487/Aiisha-Matsungo-IoT-Project.git`
2. `pip install -r requirements.txt`
3. `docker-compose up -d`
4. `python3 generate_all_sensors.py`
5. Import `pipeline/flow.json` into Node-RED at http://localhost:1880
6. Open Grafana at http://localhost:3000 (admin/admin)

### ML Pipeline (Part B)
1. `python3 step0_download.py`
2. `python3 step2_train.py` — RF accuracy: 0.926
3. `python3 step4_compare_models.py` — SVM accuracy: 0.962, 27 kB
4. `python3 step6_drift_sim.py` — drift to 0.738 at scale=0.75
5. `python3 step7_retrain.py` — recovery to 0.975

## Sensors
| Sensor | Purpose | Interface |
|--------|---------|-----------|
| MAX9814 (custom chip) | Sound pressure level | Analog (ADC) |
| DHT22 | Temperature + humidity | Digital |
| LDR Module | Light / occupancy proxy | Analog (ADC) |

## AI Disclosure
Claude (Anthropic) was used for data generation scripts, Flux queries, LaTeX report drafting. All ML scripts are Eduardo Baena's seminar code. All outputs were personally verified by running scripts and confirming results.
