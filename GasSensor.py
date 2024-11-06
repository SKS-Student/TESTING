import time
import spidev

# Initialize the SPI bus
spi = spidev.SpiDev()
spi.open(0, 0)  # Open SPI bus 0, device 0
spi.max_speed_hz = 50000  # Set the SPI speed

# Function to read analog value from the MCP3008 ADC
def read_adc(channel):
    if channel < 0 or channel > 7:
        return -1
    # Send the start bit, the channel, and the dummy byte (0)
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    # Combine the two bytes of the ADC result
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# MQ-2 sensor is connected to channel 0 of the ADC
mq_channel = 0

while True:
    # Read analog data from the MQ sensor
    sensor_value = read_adc(mq_channel)
    
    # Convert the ADC value to a more useful range (this will depend on the sensor)
    print(f'MQ-2 Sensor value: {sensor_value}')

    time.sleep(2)  # Wait for 2 seconds before reading again
