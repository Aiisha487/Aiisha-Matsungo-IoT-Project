# Acoustic Monitoring System — EECE 5155 E1

**Student:** Aiisha Matsungo  
**NUID:** 002530298  
**Course:** EECE 5155 Wireless Sensor Networks and IoT Systems  
**Semester:** Spring 2026  
**Wokwi Project:** https://wokwi.com/projects/461101098635750401  

## Project Description
An ESP32-based acoustic monitoring system simulated in Wokwi 
for quiet zone compliance in indoor spaces (Snell Library, 
Northeastern University). The system reads sound level, 
light level, temperature, and humidity, publishing structured 
JSON via MQTT to broker.hivemq.com and HTTP to ThingSpeak.

## Sensors
| Sensor | Pin | Type | Purpose |
|--------|-----|------|---------|
| Potentiometer | GPIO 34 | Native | MAX9814 microphone proxy |
| DHT22 | GPIO 33 | Native | Temperature and humidity |
| LDR Module | GPIO 32 | Native | Light level / occupancy proxy |
| MAX9814 | SIG | Custom chip | Sound level (documented) |

## Repository Structure
1. Open Wokwi project at link above
2. Hit Play
3. Open Serial Monitor to verify sensor output
4. Subscribe to `eece5155/Aiisha/sensors` on broker.hivemq.com

## Data Pipeline
ESP32 → WiFi → MQTT (broker.hivemq.com) → eece5155/Aiisha/sensors  
ESP32 → WiFi → HTTP → ThingSpeak Channel 3332763

## AI Disclosure
Claude (Anthropic) was used for firmware generation and 
report writing. All code was verified by running in Wokwi 
and confirming ThingSpeak responses.
