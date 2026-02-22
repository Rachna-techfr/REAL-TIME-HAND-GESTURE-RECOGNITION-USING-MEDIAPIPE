
# collect_gesture_data.py
# This script captures hand gesture data using MediaPipe and saves it to a CSV file.
import cv2
import mediapipe as mp
import csv
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Folder to store CSVs
if not os.path.exists("gesture_data"):
    os.makedirs("gesture_data")

# Ask user for gesture name
gesture_name = input("Enter gesture name (e.g., chrome, youtube, vscode, notepad, calc, keyboard): ").strip()
file_path = f"gesture_data/{gesture_name}.csv"

# Open CSV file
csv_file = open(file_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)

# Write header (21 landmarks × (x,y,z))
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
csv_writer.writerow(header)

# Start video capture
cap = cv2.VideoCapture(0)
print("Press 'q' to stop collecting data.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            row = []
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]

            # Save to CSV
            csv_writer.writerow(row)

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"✅ Data saved in {file_path}")
