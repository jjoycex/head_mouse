#define MIC_CLK_PIN  7
#define MIC_DATA_PIN 6
#define SDA_PIN 8
#define SCL_PIN 9
// Sensor I2C Address (0x68 if SDO is GND, 0x69 if SDO is 3.3V)
uint8_t i2cAddress = 0x68; // 0x68
#define DELTA_C 20
#define CALIBRATION_CYCLES 100 
#define MOUSE_NAME "ESP32 Mouse Controller"