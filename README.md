
# Robotic Desk Assistant: Vision-Guided Voice-Controlled System

This project implements a robotic desk assistant using the MyCobot 280 robotic arm. It combines vision-based object detection (YOLO), marker localization (STag), and natural language voice commands (OpenAI GPT & Whisper) to perform real-world robotic tasks such as picking up and placing objects, guided by user instructions.

## 📁 Project Structure

| File | Description |
|------|-------------|
| `camera_detect.py` | Main controller for vision-based detection and robot motion execution. Integrates STag markers and YOLO object detection. |
| `camera_callibration.py` | Handles camera calibration and Eye-in-Hand transformation matrix calculation. |
| `uvc_camera.py` | Captures video frames using UVC-compatible cameras. |
| `marker_utils.py` | Utility functions for STag marker pose estimation using OpenCV. |
| `voice_2.py` | Voice-controlled assistant. Uses wake-word detection, audio transcription (Whisper), and GPT-powered intent parsing to command the robot. |
| `camera_params.npz` | NumPy file containing the camera matrix (`mtx`) and distortion coefficients (`dist`). |
| `EyesInHand_matrix.json` | JSON file storing the Eye-in-Hand transformation matrix calculated during calibration. |

## 🦾 Features

- **Voice Control (Hands-Free Interaction):**
  - Uses Porcupine wake-word detection.
  - Transcribes voice using OpenAI Whisper.
  - Determines intent using GPT.
  - Converts responses to speech using OpenAI TTS.

- **Object Detection:**
  - Real-time YOLOv8 object detection on selected classes.
  - STag marker-based 6DoF pose estimation inside YOLO bounding boxes.

- **Robotic Actions:**
  - Vision-based pick-and-place with spatial transformation via Eye-in-Hand calibration.
  - Directional movement logic (`pick`, `to_me`, `left`, `right`) defined by voice.

- **Camera Calibration:**
  - Supports loading/saving camera calibration.
  - Eye-in-Hand matrix calibration using multiple poses and PnP calculations.

## 🔧 Installation

1. **Hardware Requirements:**
   - MyCobot 280 (connected via USB)
   - UVC-compatible webcam
   - Host PC (tested on Linux)
   - Microphone and speaker

2. **Python Dependencies:**

```bash
pip install opencv-python numpy scipy pyserial pyaudio pvporcupine sounddevice openai python-dotenv ultralytics
```

3. **Environment Variables:**
   Create a `.env` file with your keys:

```
OPENAI_API_KEY=your_openai_key
PORCUPINE_ACCESS_KEY=your_porcupine_key
```

4. **Model Files:**
   - Download and place `yolov8s.pt` in the project directory.
   - Configure `keyword_paths` in `voice_2.py` to point to your `.ppn` wake word file.

## 🚀 How to Run

### 1. Calibrate Camera & Arm:
```bash
python camera_callibration.py
```
This saves the Eye-in-Hand transformation matrix in `EyesInHand_matrix.json`.

### 2. Voice Assistant:
```bash
python voice_2.py
```
Say the wake word (e.g., "Hey Nova") followed by a command like:
> "Pick up the bottle and move it to the left"

### 3. Manual Execution:
You can manually test vision-based movement using:
```python
from camera_detect import camera_detect
# Load calibration data
params = np.load("camera_params.npz")
m = camera_detect(0, 43, params["mtx"], params["dist"], mc)
m.vision_trace_loop_yolo(mc, [39], "pick")  # 39 = bottle class
```

## 🧠 Supported Voice Commands

| Example | Action |
|--------|--------|
| "Pick up the book" | Pick object and return to origin |
| "Pick up the laptop and move left" | Pick and place to the left |
| "Bring the cup to me" | Deliver to home position |
| "Drop object" | Deactivate suction |
| "Move arm to home" | Predefined pose change |

## ⚠️ Known Issues

- Requires stable lighting for marker and object detection.
- Wake-word model file path must be correctly set in `voice_2.py`.
- Arduino port detection may fail on some machines; manually configure if needed.

## 📸 Demo Setup

- Recommended camera resolution: **640x480**
- Marker size used: **43mm**
- Supported YOLO object classes: [bottle, book, cup, laptop, etc.]

## 👨‍💻 Authors

- **Devang Gupta** – Final Year Project, City University of Hong Kong

