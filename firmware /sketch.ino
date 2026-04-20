#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>

// ── Pin Definitions ──────────────────────────────────────
#define MIC_PIN  34   // Potentiometer (MAX9814 proxy)
#define LDR_PIN  32   // LDR module AO
#define DHTPIN   15   // DHT22 data pin
#define DHTTYPE  DHT22

// ── WiFi ─────────────────────────────────────────────────
#define WIFI_SSID "Wokwi-GUEST"
#define WIFI_PASS ""

// ── MQTT ─────────────────────────────────────────────────
#define MQTT_BROKER "broker.hivemq.com"
#define MQTT_PORT   1883
#define MQTT_TOPIC  "eece5155/Aiisha/sensors"

// ── Objects ──────────────────────────────────────────────
WiFiClient   espClient;
PubSubClient mqtt(espClient);
DHT          dht(DHTPIN, DHTTYPE);

void connectWiFi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" Connected!");
}

void connectMQTT() {
  while (!mqtt.connected()) {
    Serial.print("Connecting to MQTT...");
    String clientId = "esp32-" + String(random(0xffff), HEX);
    if (mqtt.connect(clientId.c_str())) {
      Serial.println(" Connected!");
    } else {
      Serial.printf(" Failed (rc=%d). Retrying in 5s...\n", mqtt.state());
      delay(5000);
    }
  }
}

void publishSensorData() {
  // 1. Sound level (potentiometer proxy for MAX9814)
  int micRaw    = analogRead(MIC_PIN);
  int soundLevel = map(micRaw, 0, 4095, 0, 100);

  // 2. Light level (LDR module)
  int ldrRaw    = analogRead(LDR_PIN);
  int lightLevel = map(ldrRaw, 0, 4095, 0, 100);

  // 3. Temperature + Humidity (DHT22)
  float temp = dht.readTemperature();
  float hum  = dht.readHumidity();

  if (isnan(temp) || isnan(hum)) {
    Serial.println("ERROR: DHT22 read failed!");
    temp = -1;
    hum  = -1;
  }

  // Structured JSON
  String json = "{";
  json += "\"device_id\":\"esp32-node-01\",";
  json += "\"timestamp\":"  + String(millis()) + ",";
  json += "\"sensors\":{";
  json += "\"sound\":{\"raw\":"  + String(micRaw)
        + ",\"level\":"          + String(soundLevel)
        + ",\"unit\":\"%\"},";
  json += "\"light\":{\"raw\":"  + String(ldrRaw)
        + ",\"level\":"          + String(lightLevel)
        + ",\"unit\":\"%\"},";
  json += "\"temperature\":{\"value\":" + String(temp, 1)
        + ",\"unit\":\"C\"},";
  json += "\"humidity\":{\"value\":"    + String(hum, 1)
        + ",\"unit\":\"%\"}";
  json += "}}";

  Serial.println("Publishing: " + json);
  mqtt.publish(MQTT_TOPIC, json.c_str());
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  connectWiFi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqtt.connected())             connectMQTT();
  mqtt.loop();
  publishSensorData();
  delay(20000);
}
