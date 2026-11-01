#define EIDSP_QUANTIZE_FILTERBANK   0
#include <xuxw76-project-1_inferencing.h>
#include "driver/i2s_pdm.h"
#include "config.h"
#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"
#include <ArduinoEigenDense.h>
#include <Filters.h>
#include <HijelHID_BLEMouse.h>

using namespace Eigen;

i2s_chan_handle_t rx_handle; // For audio recording
BMI270 imu;

HijelBLEMouse bleMouse(MOUSE_NAME, "Espressif", 100);

Matrix3f R_ag;
FilterOnePole lowpassX(LOWPASS, 10);
FilterOnePole lowpassY(LOWPASS, 10);
int16_t sample_buffer[16000];

void setup() {
  Serial.begin(9600);

  Serial.println("Starting BLE Mouse...");  
  bleMouse.setLogLevel(HIDLogLevel::Normal); // optional: prints pairing/connection events too
  bleMouse.begin();

  // 1. Create a channel config using the generic i2s_new_channel function
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  // Allocate the RX channel
  i2s_new_channel(&chan_cfg, NULL, &rx_handle);
  // 2. Configure PDM RX parameters
  i2s_pdm_rx_config_t pdm_rx_cfg = {
    .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(16000), // 16 kHz sample rate
    .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
    .gpio_cfg = {
      .clk = (gpio_num_t)MIC_CLK_PIN,
      .din = (gpio_num_t)MIC_DATA_PIN,
      .invert_flags = {
        .clk_inv = false,
      },
    },
  };
  // 3. Initialize and enable PDM RX mode
  i2s_channel_init_pdm_rx_mode(rx_handle, &pdm_rx_cfg);
  i2s_channel_enable(rx_handle);

    // Initialize I2C with ESP32-S3 pins
  Wire.begin(SDA_PIN, SCL_PIN);

  // Begin communication with BMI270
  while (imu.beginI2C(i2cAddress, Wire) != 0) {
    Serial.println("Error: BMI270 not detected. Check connections!");
    delay(1000);
  }

  Serial.println("BMI270 successfully initialized! Start calibration");
  Vector3f a_bar(0, 0, 0);
  for (int i=0; i<CALIBRATION_CYCLES; i++) {
    imu.getSensorData();
    a_bar(0) += imu.data.accelX;
    a_bar(1) += imu.data.accelY;
    a_bar(2) += imu.data.accelZ;
    delay(DELTA_C);
  }

  a_bar /= (float)(CALIBRATION_CYCLES);
  R_ag = calibrate_with_g(a_bar);
  Serial.println("BMI270 Calibration is done");
  run_classifier_init(); 
}

Matrix3f get_cross_matrix(Vector3f u) {
  Matrix3f m = Matrix3f::Zero(3, 3);
  m(0, 1) = -u(2); m(0, 2) = u(1); m(1, 2) = -u(0);
  m(1, 0) = u(2); m(2, 0) = -u(1); m(2, 1) = u(0);
  return m;
}

Matrix3f calibrate_with_g(Vector3f a_bar) {
  Vector3f g(1, 0, 0);
  Vector3f a_hat = a_bar.normalized();
  float theta = acos(a_hat.dot(g));
  Serial.printf("a_bar: [%f, %f, %f]\n", a_bar(0), a_bar(1), a_bar(2));
  Serial.printf("theta: %f\n", theta);

  Matrix3f k_x = get_cross_matrix(a_hat.cross(g));

  return Matrix3f::Identity(3, 3) +  sin(theta)*k_x + (1-cos(theta))*(k_x*k_x);
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&sample_buffer[offset], out_ptr, length);
    return 0;
}

void loop() {
  size_t bytes_read = 0;
  static size_t bytes_stored = 0;
  static long last_click_ms = 0;
  static const size_t MAX_READ_SIZE = 512;
  static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
  // Read data from PDM mic
  if (i2s_channel_read(rx_handle, sample_buffer + bytes_stored / 2,
       min(MAX_READ_SIZE, sizeof(sample_buffer) - bytes_stored), &bytes_read, 1000) == ESP_OK) {
    int samples_read = bytes_read / sizeof(int16_t);
    if (samples_read > 0 &&  (millis() - last_click_ms) > 1000)  {
      int16_t max_volume = 0;
      for (int i = 0; i < samples_read; ++i) 
          max_volume= max(max_volume, sample_buffer[i]);
      if (max_volume > 250) {
        Serial.printf("max_volume: %d\n", max_volume);
        last_click_ms = millis();
        bleMouse.click(MouseButton::Left);
      }
    }
  }

  // Read raw measurements from gyro
  imu.getSensorData();
  Vector3f omega(imu.data.gyroX, imu.data.gyroY, imu.data.gyroZ);
  Vector3f omega_1 = R_ag*omega;
  lowpassX.input(omega_1(0));
  lowpassY.input(omega_1(2));
  signed char dx = 0.50*lowpassX.output();
  signed char dy = 0.25*lowpassY.output();
  if (abs(dx) > 64) { dx = (dx/abs(dx))*64; }
  if (abs(dy) > 64) { dy = (dy/abs(dy))*64; }
  bleMouse.move(dx, -dy);
}
