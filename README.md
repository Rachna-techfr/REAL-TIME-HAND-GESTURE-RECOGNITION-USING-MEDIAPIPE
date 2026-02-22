# REAL-TIME-HAND-GESTURE-RECOGNITION-USING-MEDIAPIPE
# ✋ Real-Time Hand Gesture Recognition System (MediaPipe + KNN)

## 📌 Overview

This project is a real-time hand gesture recognition and computer control system built using MediaPipe, OpenCV, and Machine Learning (KNN).

The system allows users to control their computer using hand gestures detected through a webcam. It supports mouse control, application launching, app-specific actions, virtual keyboard interaction, and voice feedback.

The system uses MediaPipe for 21-point hand landmark detection and a trained KNN model for custom gesture classification.

---

## 🚀 Features

### 🖱️ Mouse Control (Normal Mode)
- Cursor movement using index finger
- Pinch gesture → Left click
- Middle-thumb gesture → Right click
- Vertical finger movement → Scroll
- Fist gesture → Drag & drop

### 💻 Application Control (App Mode)
Using gestures to open applications:
- Google Chrome
- YouTube
- Notepad
- Calculator
- VS Code
- Virtual Keyboard

### 🎯 App-Specific Controls

Chrome:
- Victory → New Tab
- Rock → Scroll Down
- Thumbs Up → Refresh

YouTube:
- Victory → Play/Pause
- Rock → Volume Up
- Thumbs Up → Next Video

VS Code:
- Victory → Save File
- Rock → Run Code
- Thumbs Up → Open Terminal

### ⌨️ Virtual Keyboard
- Controlled using pinch detection
- On-screen keyboard built using Tkinter
- Supports typing, backspace, clear, and space

### 🤖 Machine Learning Integration
- Custom gesture dataset collection
- Dataset merging
- KNN classifier training
- Real-time ML-based gesture prediction
- Model saved using Joblib

### 🔊 Voice Feedback
- Text-to-speech feedback using pyttsx3
- Announces app launches and actions

---

## 🧠 Project Workflow

### Step 1: Collect Gesture Data

Run:
python collect_gesture_data.py

- Captures 21 hand landmarks
- Saves gesture data into CSV files inside gesture_data folder

---

### Step 2: Combine Dataset

Run:
python combining_data.py

- Merges all gesture CSV files
- Creates gesture_dataset.csv

---

### Step 3: Train Model

Run:
python train_knn_model.py

- Splits dataset into train/test
- Trains KNN classifier
- Prints accuracy
- Saves trained model as gesture_knn_model.pkl

---

### Step 4: Run Gesture Interface

Run:
python all.py

- Starts real-time webcam interface
- Enables gesture-based system control

Press ESC to exit.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn (KNN)
- PyAutoGUI
- Tkinter
- Pyttsx3
- Joblib

---

## 📁 Project Structure

collect_gesture_data.py  
combining_data.py  
train_knn_model.py  
all.py  
gesture_knn_model.pkl  
gesture_dataset.csv  
gestures.json  
requirements.txt  
gesture_data/  

---

## ⚙️ Installation

1. Clone the repository:

git clone https://github.com/your-username/your-repository-name.git  
cd your-repository-name  

2. Create virtual environment (optional but recommended):

python -m venv venv  
venv\Scripts\activate  

3. Install dependencies:

pip install -r requirements.txt  

4. Run the project:

python all.py  

---

## 🎯 Applications

- Touchless computer interaction
- Accessibility tools
- Smart classroom systems
- Human-computer interaction research
- AI-based UI control systems

---

## 📈 Future Improvements

- Deep learning-based gesture classification
- Multi-hand tracking
- Cross-platform support
- Gesture customization interface
- Web-based version
- Performance optimization

---

## 👩‍💻 Author
**Rachna R**  
B.Voc (AI&ML)
