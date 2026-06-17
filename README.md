# Cyanora_Qingyou_pi5

A lightweight multimodal emotion recognition system designed for Raspberry Pi 5. It fuses facial expressions, camera-based heart rate (rPPG), and smartwatch BLE heart rate data to detect emotions and play soothing music automatically.

## ️ Hardware Requirements

- **Raspberry Pi 5** (4GB+ RAM recommended)
- **Camera Module** (e.g., Camera Module 3)
- **Smartwatch / Fitness Band** (with BLE Heart Rate broadcast)
- **Active Cooling** (Required for continuous AI processing)

##  Quick Setup

1. **Flash OS**: Use Raspberry Pi Imager to flash **Raspberry Pi OS (64-bit)**.
2. **Clone & Install**:
   ```bash
   git clone [https://github.com/Enola-Eve/Cyanora_Qingyou_pi5.git]
   cd Cyanora_Qingyou_pi5
   chmod +x install.sh && ./install.sh
3. **run**:
   python emotion.py

## Project structure

Cyanora_Qingyou_pi5/
├── emotion2.py           # Main application
├── emotion_model.tflite # Lightweight emotion model
├── install.sh           # One-click setup script
├── requirements.txt     # Python dependencies
└── music/               # Soothing music files (.mp3)

## Tips

Lighting: Ensure even facial lighting for accurate rPPG heart rate detection.
Thermal: Monitor CPU temp. If it exceeds 80°C, performance will throttle.
Bluetooth: Keep the smartwatch close to the Pi for stable BLE connection.

## Licence

MIT Licence
