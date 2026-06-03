const int PUMP_PIN = 7;

void setup() {
  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') {
      digitalWrite(PUMP_PIN, HIGH);
    } else if (cmd == '0') {
      digitalWrite(PUMP_PIN, LOW);
    }
  }
}