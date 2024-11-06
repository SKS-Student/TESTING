import spidev
import time

# Initialize the SPI bus
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, Device 0
spi.max_speed_hz = 50000  # Set SPI speed

# Function to read from MCP3008
def read_adc(channel):
    """Read data from the ADC channel (0-7) on MCP3008"""
    if channel < 0 or channel > 7:
        return -1  # Invalid channel
    # Send the start bit and the channel to the ADC
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    # Combine the two bytes of the ADC result
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# Set the channel for soil moisture sensor (A0 -> CH0 on MCP3008)
soil_moisture_channel = 0

while True:
    # Read moisture data from the soil sensor
    moisture_level = read_adc(soil_moisture_channel)
    
    # Print the moisture level
    print(f'Soil Moisture Level (ADC value): {moisture_level}')
    
    # Convert the ADC value (0-1023) to percentage (0% - 100%)
    moisture_percentage = (moisture_level / 1023) * 100
    print(f'Soil Moisture Level: {moisture_percentage:.2f}%')
    
    # Provide feedback based on moisture level (optional)
    if moisture_percentage < 30:
        print("Warning: Soil is dry. Consider watering the plants!")
    elif moisture_percentage > 70:
        print("Soil is wet. No watering needed.")
    else:
        print("Soil moisture is in optimal range.")
    
    # Wait 2 seconds before reading again
    time.sleep(2)

# moisture_percentage = (moisture_level / 1023) * 100
