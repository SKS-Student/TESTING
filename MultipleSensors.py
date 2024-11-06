import time
import Adafruit_DHT
import board
import adafruit_bme280.legacy
import spidev

# Setup for DHT22
dht_sensor = Adafruit_DHT.DHT22
dht_pin = 4

# Setup for BME280
i2c = board.I2C()
bme280 = adafruit_bme280.legacy.Adafruit_BME280_I2C(i2c)

# Setup for MQ-2
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 50000

def read_adc(channel):
    if channel < 0 or channel > 7:
        return -1
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

mq_channel = 0  # MQ-2 connected to channel 0

while True:
    # DHT22 Readings
    humidity, temperature = Adafruit_DHT.read_retry(dht_sensor, dht_pin)
    if humidity is not None and temperature is not None:
        print(f'DHT22 -> Temperature: {temperature:.1f}C, Humidity: {humidity:.1f}%')
    else:
        print('Failed to read DHT22 sensor')

    # BME280 Readings
    temperature_bme = bme280.temperature
    humidity_bme = bme280.humidity
    pressure_bme = bme280.pressure
    print(f'BME280 -> Temperature: {temperature_bme:.2f} °C, Humidity: {humidity_bme:.2f} %, Pressure: {pressure_bme:.2f} hPa')

    # MQ-2 Readings
    mq_value = read_adc(mq_channel)
    print(f'MQ-2 -> Sensor value: {mq_value}')

    time.sleep(2)  # Wait for 2 seconds before reading again
