#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <FirebaseESP8266.h>
#include <time.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ======================================================
// LCD Configuration (16x2, address 0x27 or 0x3F)
// ======================================================
LiquidCrystal_I2C lcd(0x27, 16, 2);   // try 0x3F if 0x27 doesn't work

// ======================================================
// Wi-Fi & Firebase
// ======================================================
#define WIFI_SSID     "Airtel_sahi_0849"
#define WIFI_PASSWORD "air99772"

#define API_KEY       "AIzaSyACk3Kj9agZgUhYmEKGVt0Ul7avCJSoNxY"
// IMPORTANT: No https:// and no trailing slash
#define DATABASE_URL  "aquaculture-af54d-default-rtdb.firebaseio.com"
#define USER_EMAIL    "sb284160@gmail.com"
#define USER_PASSWORD "Password@1"

// ======================================================
// Objects
// ======================================================
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;
ESP8266WebServer server(80);

// ======================================================
// Sensor Pins & Variables
// ======================================================
#define TURBIDITY_PIN A0

int cleanWaterValue = 0;
float pH = 0.0;
float temp = 0.0;
int turbidityPercent = 0;
String incomingData = "";
unsigned long lastPrintTime = 0;
unsigned long lastFirebasePush = 0;
unsigned long lastDisplayUpdate = 0;
const unsigned long FIREBASE_INTERVAL = 5000;
const unsigned long DISPLAY_INTERVAL = 1000;

// ======================================================
// Turbidity Calibration
// ======================================================
void calibrateTurbidity() {
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Calibrating");
  lcd.setCursor(0,1);
  lcd.print("Turbidity...");
  delay(2000);

  Serial.println();
  Serial.println("===============================");
  Serial.println(" TURBIDITY CALIBRATION");
  Serial.println("===============================");
  Serial.println("Place sensor in CLEAN WATER");
  Serial.println("Calibration starts in 5 seconds...");
  delay(5000);

  long sum = 0;
  for (int i = 0; i < 200; i++) {
    sum += analogRead(TURBIDITY_PIN);
    delay(20);
  }
  cleanWaterValue = sum / 200;
  Serial.print("Calibration Complete");
  Serial.print("\nClean Water Raw Value = ");
  Serial.println(cleanWaterValue);
  Serial.println("===============================");

  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Calibration OK");
  delay(1000);
}

void readTurbidity() {
  int raw = analogRead(TURBIDITY_PIN);
  turbidityPercent = map(raw, cleanWaterValue, 0, 0, 100);
  turbidityPercent = constrain(turbidityPercent, 0, 100);
  if (turbidityPercent <= 2) turbidityPercent = 0;
}

// ======================================================
// Parse Serial Data from Nano
// ======================================================
void parseSerialData(String data) {
  Serial.print("[RAW] ");
  Serial.println(data);

  float phVal = 0, tempVal = 0;
  bool found = false;

  int phStart = data.indexOf("pH");
  if (phStart != -1) {
    int colonPos = data.indexOf(':', phStart);
    if (colonPos == -1) colonPos = data.indexOf(' ', phStart);
    if (colonPos != -1) {
      int start = colonPos + 1;
      while (start < data.length() && data[start] == ' ') start++;
      int end = start;
      while (end < data.length() && (isDigit(data[end]) || data[end] == '.')) end++;
      if (end > start) {
        phVal = data.substring(start, end).toFloat();
        found = true;
      }
    }
  }

  int tempStart = data.indexOf("Temp");
  if (tempStart == -1) tempStart = data.indexOf("temperature");
  if (tempStart != -1) {
    int colonPos = data.indexOf(':', tempStart);
    if (colonPos == -1) colonPos = data.indexOf(' ', tempStart);
    if (colonPos != -1) {
      int start = colonPos + 1;
      while (start < data.length() && data[start] == ' ') start++;
      int end = start;
      while (end < data.length() && (isDigit(data[end]) || data[end] == '.')) end++;
      if (end > start) {
        tempVal = data.substring(start, end).toFloat();
        found = true;
      }
    }
  }

  if (found) {
    pH = phVal;
    temp = tempVal;
    Serial.print("[Nano] pH = ");
    Serial.print(pH, 2);
    Serial.print(" | Temp = ");
    Serial.println(temp, 1);
  } else {
    Serial.println("[Nano] Could not parse data.");
  }
}

void readSerialFromNano() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      parseSerialData(incomingData);
      incomingData = "";
    } else if (c != '\r') {
      incomingData += c;
    }
  }
}

// ======================================================
// Get Unix Timestamp via NTP
// ======================================================
unsigned long getTimestamp() {
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  time_t now = time(nullptr);
  while (now < 100000) {
    delay(100);
    now = time(nullptr);
  }
  return now;
}

// ======================================================
// ✅ CORRECTED: Push Sensor Data to Firebase
// ======================================================
void pushToFirebase() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi not connected. Skipping Firebase push.");
    return;
  }

  unsigned long timestamp = getTimestamp();

  // Use FirebaseJson object (not a string)
  FirebaseJson json;
  json.set("ph", pH);
  json.set("temperature", temp);
  json.set("turbidity", turbidityPercent);
  json.set("timestamp", timestamp);

  // Correct path: no .json, no trailing slash
  String path = "/aquaculture/sensors";

  Serial.print("Firebase path: ");
  Serial.println(path);
  String jsonStr;
  json.toString(jsonStr);
  Serial.print("Payload: ");
  Serial.println(jsonStr);

  if (Firebase.setJSON(fbdo, path, json)) {
    Serial.println("✅ Firebase update successful.");
  } else {
    Serial.print("❌ Firebase update failed: ");
    Serial.println(fbdo.errorReason());
  }
}

// ======================================================
// Update 16x2 LCD
// ======================================================
void updateDisplay() {
  lcd.setCursor(0, 0);
  lcd.print("pH:");
  lcd.print(pH, 2);
  lcd.print(" T:");
  lcd.print(temp, 1);
  lcd.print((char)223);
  lcd.print("C");
  lcd.print("   ");

  lcd.setCursor(0, 1);
  lcd.print("Turb:");
  lcd.print(turbidityPercent);
  lcd.print("%");
  lcd.print(" ");
  if (turbidityPercent <= 20)      lcd.print("CLEAR");
  else if (turbidityPercent <= 50) lcd.print("CLOUDY");
  else                             lcd.print("DIRTY");
  lcd.print("    ");
}

// ======================================================
// Web Server – Root (JSON)
// ======================================================
void handleRoot() {
  String json = "{\"pH\":" + String(pH, 2) +
                ",\"temperature\":" + String(temp, 1) +
                ",\"turbidity\":" + String(turbidityPercent) +
                ",\"voltage\":" + String(analogRead(TURBIDITY_PIN) * (3.3 / 1023.0), 2) +
                ",\"status\":\"" +
                (turbidityPercent <= 20 ? "CLEAR" : (turbidityPercent <= 50 ? "CLOUDY" : "DIRTY")) +
                "\"}";
  server.send(200, "application/json", json);
}

// ======================================================
// Setup
// ======================================================
void setup() {
  Serial.begin(9600);
  delay(1000);

  // ---- Initialize 16x2 LCD ----
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("ESP8266 Aqua");
  lcd.setCursor(0,1);
  lcd.print("Initializing...");
  delay(2000);

  calibrateTurbidity();

  // ---- Connect to Wi-Fi ----
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP address: ");
  Serial.println(WiFi.localIP());

  // ---- Configure Firebase (Mobizt library) ----
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;      // no https://
  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;
  Firebase.reconnectWiFi(true);

  Serial.print("Authenticating to Firebase...");
  Firebase.begin(&config, &auth);

  // Wait for authentication (using Firebase.ready() is safer)
  unsigned long start = millis();
  while (!Firebase.ready() && millis() - start < 10000) {
    delay(100);
    Serial.print(".");
  }
  if (Firebase.ready()) {
    Serial.println(" Authenticated!");
    Serial.print("User UID: ");
    Serial.println(auth.token.uid.c_str());
  } else {
    Serial.println(" Authentication failed!");
  }

  // ---- Start Web Server ----
  server.on("/", handleRoot);
  server.begin();
  Serial.println("HTTP server started.");

  // ---- Show IP on LCD ----
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("IP:");
  lcd.print(WiFi.localIP());
  lcd.setCursor(0,1);
  lcd.print("Ready!");
  delay(3000);
  lcd.clear();
}

// ======================================================
// Loop
// ======================================================
void loop() {
  readSerialFromNano();
  readTurbidity();

  // Print combined data every 3 seconds
  if (millis() - lastPrintTime > 3000) {
    lastPrintTime = millis();
    Serial.println("========== ESP8266 Combined Data ==========");
    Serial.print("Turbidity: ");
    Serial.print(turbidityPercent);
    Serial.println("%");
    Serial.print("pH (from Nano): ");
    Serial.println(pH, 2);
    Serial.print("Temp (from Nano): ");
    Serial.println(temp, 1);
    Serial.println("===========================================");
  }

  // Update LCD every second
  if (millis() - lastDisplayUpdate > DISPLAY_INTERVAL) {
    lastDisplayUpdate = millis();
    updateDisplay();
  }

  // Push to Firebase every 5 seconds
  if (millis() - lastFirebasePush > FIREBASE_INTERVAL) {
    lastFirebasePush = millis();
    pushToFirebase();
  }

  server.handleClient();
  delay(50);
}