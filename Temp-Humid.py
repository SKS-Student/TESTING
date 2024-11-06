import Adafruit_DHT
import time

# Set the sensor type and the GPIO pin it's connected to
sensor = Adafruit_DHT.DHT22  # or Adafruit_DHT.DHT11 for the DHT11 sensor
pin = 4  # GPIO pin where the sensor is connected

while True:
    # Attempt to read the sensor
    humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)

    if humidity is not None and temperature is not None:
        print(f'Temperature: {temperature:.1f}C')
        print(f'Humidity: {humidity:.1f}%')
    else:
        print('Failed to retrieve data from sensor')

    time.sleep(2)  # Wait for 2 seconds before reading again


# with presssure sensor
import time
import board
import adafruit_bme280.legacy

# Initialize the I2C bus and the BME280 sensor
i2c = board.I2C()  # Uses the default I2C bus
bme280 = adafruit_bme280.legacy.Adafruit_BME280_I2C(i2c)

while True:
    # Read sensor data
    temperature = bme280.temperature
    humidity = bme280.humidity
    pressure = bme280.pressure

    print(f'Temperature: {temperature:.2f} °C')
    print(f'Humidity: {humidity:.2f} %')
    print(f'Pressure: {pressure:.2f} hPa')

    time.sleep(2)  # Wait for 2 seconds before reading again
