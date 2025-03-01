import time
import board
import busio
import adafruit_bno055
import matplotlib.pyplot as plt
from collections import deque

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

x_data, y_data, z_data = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)

plt.ion()
fig, ax = plt.subplots()

while True:
    x, y, z = sensor.euler  # Read Roll, Pitch, Yaw
    
    if x is not None and y is not None and z is not None:
        x_data.append(x)
        y_data.append(y)
        z_data.append(z)
    
    ax.clear()
    ax.plot(x_data, label="Roll (X)")
    ax.plot(y_data, label="Pitch (Y)")
    ax.plot(z_data, label="Yaw (Z)")
    ax.legend()
    
    plt.pause(0.1)  

