# Pixhawk Flight Controller Calibration using QGroundControl

## Overview
This guide explains how to calibrate your **Pixhawk 2.4.8** flight controller to fly the drone using **QGroundControl (QGC)**. Calibration ensures proper sensor alignment and stable flight.

## Prerequisites
### Hardware Components:
- **Pixhawk 2.4.8 Flight Controller**
- **Raspberry Pi 5 (8GB) (for processing)**
- **A2212 2200KV Brushless Motors** (x4)
- **30A ESCs** (Electronic Speed Controllers) (x4)
- **5200mAh 3S LiPo Battery**
- **FS-i6 Transmitter & FS-iA6B Receiver** (for manual control)
- **PX4 Optical Flow Sensor & BNO055 IMU** (for stabilization)

### Software Requirements:
- **QGroundControl (QGC)** (Download from [QGroundControl Website](https://qgroundcontrol.com))
- **Latest PX4 Firmware** (Flashed onto Pixhawk)

---

## Step-by-Step Calibration

### 1. Install QGroundControl
- Download and install QGC from the official website.
- Connect the Pixhawk to your PC via **Micro USB**.

### 2. Connect the Pixhawk to QGC
- Open QGroundControl and wait for it to detect the Pixhawk.
- If not detected, check your USB cable and driver installation.

### 3. Calibrate Accelerometer
- Navigate to **Sensors > Accelerometer Calibration**.
- Follow the on-screen instructions:
  1. Place the drone level.
  2. Rotate to each requested position.
  3. Ensure the calibration completes successfully.

### 4. Calibrate Gyroscope
- Go to **Sensors > Gyroscope Calibration**.
- Keep the drone still while calibration runs.

### 5. Calibrate Magnetometer (Compass)
- Navigate to **Sensors > Compass Calibration**.
- Rotate the drone in different orientations as instructed.
- Perform calibration away from metallic objects.

### 6. Calibrate Radio (Transmitter & Receiver)
- Ensure your **FS-i6 Transmitter & FS-iA6B Receiver** are bound.
- Navigate to **Radio Setup** in QGC.
- Move all sticks and switches to detect movement.

### 7. ESC and Motor Calibration
- Remove **propellers** before calibration.
- In QGC, navigate to **ESC Calibration**.
- Follow on-screen instructions to set throttle endpoints.
- Power cycle the drone after calibration.

### 8. Flight Mode Configuration
- Assign flight modes in **QGC > Flight Mode Setup**.
- Recommended modes:
  - **Stabilize** (Manual control with self-leveling)
  - **Loiter** (GPS position hold)
  - **Return to Home (RTH)** (Failsafe mode)

### 9. Pre-Flight Checklist
- Ensure all sensors pass calibration.
- Test **motor directions** in QGC.
- Set failsafe options for battery and signal loss.

### 10. First Flight Test
- Take off in **Stabilize Mode**.
- Verify stability before testing **Loiter** or **Auto Mode**.
- Land and check logs in **QGC > Vehicle Logs**.

## Troubleshooting
- **Calibration Fails:** Restart QGC and try again.
- **No RC Signal:** Rebind **FS-i6 Transmitter** to the **FS-iA6B Receiver**.
- **Drifting Issue:** Recalibrate accelerometer and compass.
- **Motors Not Spinning:** Verify ESC connections and power supply.

## Additional Notes
- Ensure **firmware is up to date** before calibration.
- Keep **propellers removed** during calibration to avoid accidents.
- Perform calibration in an open area away from **metal objects and interference**.

## References
- QGroundControl Documentation: [https://docs.qgroundcontrol.com](https://docs.qgroundcontrol.com)
- PX4 Autopilot Documentation: [https://docs.px4.io](https://docs.px4.io)

---

This guide ensures your **Pixhawk-based drone is fully calibrated and flight-ready** using **QGroundControl**. 🚀

