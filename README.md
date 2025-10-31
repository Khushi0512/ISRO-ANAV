# ISRO ANAV:
This Project was Developed under the IROC challenge 2024-25 Provided by ISRO, by a group of 10 students at Dharmsinh Desai University, including myself. A research paper with me as the co-author along with my teammembers and mentor has been published as well by our university. 

## Overview
ISRO ANAV is an autonomous aerial vehicle designed for precision navigation, mapping, and real-time decision-making in complex environments, including extraterrestrial terrains. The system integrates cutting-edge technologies such as SLAM, Convolutional Neural Networks (CNN), and adaptive landing gear.

## Features
- **Autonomous Navigation:** Uses sensor fusion, SLAM, and real-time path planning.
- **Landing Spot Detection:** AI-driven crater detection and pose estimation for safe landings.
- **Localization & Mapping:** LiDAR and stereo vision for detailed environmental awareness.
- **Emergency Response System:** Automated failsafe mechanisms for secure landings.
- **High-Precision Hardware:** Raspberry Pi 5, Pixhawk flight controller, and IMU for stability and processing.
- **Real-Time Communication:** 2.4GHz transmitter and telemetry module for control and monitoring.

## System Components
### Hardware
- **Aerial Vehicle:** Lightweight quadcopter drone (<2kg) with modular architecture.
- **Propulsion:** Four 2200KV brushless DC motors, 30A ESCs, and 5200mAh 3S LiPo battery.
- **Navigation Sensors:** Optical flow, 3D LiDAR, and IMU for real-time motion tracking.
- **Computation:** Raspberry Pi 5 (8GB) for onboard image processing and CNN-based decision-making.
- **Landing Gear:** Adaptive design to adjust to sloped surfaces and rough terrain.

### Software
- **SLAM (Simultaneous Localization and Mapping):** For terrain mapping and real-time localization.
- **CNN-based Image Processing:** Hazard detection and terrain recognition.
- **Failsafe Algorithms:** Automated return-to-home (RTH) and emergency landing protocols.
- **Sensor Fusion:** Kalman filtering for data accuracy and noise reduction.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/userofmeet/ISRO-ANAV.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ISRO-ANAV
   ```

## PDF Documentation
The full project documentation is available in [ANAV_Documentation.pdf](Anav_Documentation.pdf). To view it, open the file in any PDF viewer.

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```sh
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```sh
   git commit -m "Added a new feature"
   ```
4. Push to the branch:
   ```sh
   git push origin feature-branch
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](https://github.com/twbs/bootstrap/blob/main/LICENSE) for details

