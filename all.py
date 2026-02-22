# gesture_interface.py
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os
import threading
import time
import tkinter as tk
import pyttsx3
import json
import joblib

# ---------------------- Load Config (gestures.json) ----------------------
DEFAULT_CONFIG = {
    "modes": {"app_mode": "THUMBSDOWN"},
    "global_gestures": {
        "PALM": "chrome",
        "ROCK": "youtube",
        "VICTORY": "notepad",
        "FIST": "calculator",
        "OK": "vscode",
        "THUMBSUP": "keyboard"
    },
    "app_specific": {
        "chrome": {
            "VICTORY": "new_tab",
            "ROCK": "scroll_down",
            "THUMBSUP": "refresh"
        },
        "youtube": {
            "VICTORY": "play_pause",
            "ROCK": "volume_up",
            "THUMBSUP": "next_video"
        },
        "vscode": {
            "VICTORY": "save_file",
            "ROCK": "run_code",
            "THUMBSUP": "open_terminal"
        }
    },
    "thresholds": {
        "ok_distance": 0.05,
        "pinch_left_click": 0.045,
        "right_click": 0.03,
        "drag_fist": 0.10
    }
}

def load_config(path="gestures.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        merged = DEFAULT_CONFIG.copy()
        merged["modes"].update(cfg.get("modes", {}))
        merged["global_gestures"].update(cfg.get("global_gestures", {}))
        merged["app_specific"].update(cfg.get("app_specific", {}))
        merged["thresholds"].update(cfg.get("thresholds", {}))
        return merged
    except Exception as e:
        print("[INFO] Could not load gestures.json, using defaults. Error:", e)
        return DEFAULT_CONFIG

CFG = load_config()
APP_MODE_GESTURE = CFG["modes"].get("app_mode", "THUMBSDOWN")
GLOBAL_GESTURES = CFG["global_gestures"]
APP_SPECIFIC = CFG["app_specific"]
THRESH = CFG["thresholds"]

KEYBOARD_GESTURE = next((g for g, a in GLOBAL_GESTURES.items() if str(a).lower() == "keyboard"), "THUMBSUP")

# ---------------------- MediaPipe & Screen Setup ----------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# --- States & Constants -----
keyboard_open = False
app_mode = False
active_app = None
last_launch_time = 0
LAUNCH_COOLDOWN = 1.5
last_toggle_time = 0
TOGGLE_COOLDOWN = 1.0

pinch_coords = [None]
pinch_active = [False]
last_pinch_time = 0
PINCH_COOLDOWN = 0.6

smooth_x, smooth_y = 0.0, 0.0
SMOOTHING_FACTOR = 0.2
_first_detection = False

idle_mode = False
IDLE_BANNER_COLOR = (100, 100, 100)
prev_y = 0
click_down = False

gesture_log = []
MAX_LOG_LINES = 6
def add_gesture_log(action):
    global gesture_log
    timestamp = time.strftime("%H:%M:%S")
    entry = f"{timestamp} - {action}"
    gesture_log.append(entry)
    if len(gesture_log) > MAX_LOG_LINES:
        gesture_log.pop(0)

VOICE_ENABLED = True
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
except Exception:
    engine = None
    VOICE_ENABLED = False

def _tts_thread(text):
    if engine:
        engine.say(text)
        engine.runAndWait()

def speak(text):
    if VOICE_ENABLED and engine:
        threading.Thread(target=_tts_thread, args=(text,), daemon=True).start()

def feedback(text):
    add_gesture_log(text)
    speak(text)

# ---------------------- Virtual Keyboard ----------------------
def get_current_pinch_location():
    return pinch_coords[0]
def show_virtual_keyboard(get_pinch_location):
    global keyboard_open, text_var, key_buttons
    def check_pinch():
        if not keyboard_open:
            return
        pinch = get_pinch_location()
        if pinch:
            px, py = pinch
            for key, btn in key_buttons.items():
                bx1 = btn.winfo_rootx()
                by1 = btn.winfo_rooty()
                bx2 = bx1 + btn.winfo_width()
                by2 = by1 + btn.winfo_height()
                if bx1 - 10 <= px <= bx2 + 10 and by1 - 10 <= py <= by2 + 10:
                    if not pinch_active[0]:
                        current = text_var.get()
                        if key == "Space":
                            text_var.set(current + " ")
                        elif key == "Back":
                            text_var.set(current[:-1])
                        elif key == "Clear":
                            text_var.set("")
                        else:
                            text_var.set(current + key)
                        pinch_active[0] = True
                    break
        else:
            pinch_active[0] = False
        root.after(100, check_pinch)
    def close_keyboard():
        global keyboard_open
        keyboard_open = False
        root.destroy()
    root = tk.Tk()
    root.title("Gesture Virtual Keyboard")
    root.geometry("800x400")
    text_var = tk.StringVar()
    tk.Entry(root, textvariable=text_var, font=('Arial', 18),
             width=50).grid(row=0, column=0, columnspan=10, padx=10, pady=10)
    keys = [
        ['Q','W','E','R','T','Y','U','I','O','P'],
        ['A','S','D','F','G','H','J','K','L'],
        ['Z','X','C','V','B','N','M'],
        ['Space','Back','Clear']
    ]
    key_buttons = {}
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            btn = tk.Button(root, text=key, width=6, height=2)
            btn.grid(row=i+1, column=j, padx=5, pady=5)
            key_buttons[key] = btn
    tk.Button(root, text="Close", width=10, height=2, 
              command=close_keyboard).grid(row=5, column=4, columnspan=2)
    root.after(100, check_pinch)
    root.mainloop()

# ---------------------- App Launcher ----------------------
def launch_app(app_name):
    global active_app
    try:
        name = str(app_name).lower()
        if name == "notepad":
            os.system("notepad")
        elif name == "calculator":
            os.system("calc")
        elif name == "chrome":
            os.system("start chrome")
        elif name == "youtube":
            os.system("start chrome https://www.youtube.com")
        elif name == "vscode":
            os.system("code")
        elif name == "keyboard":
            if not keyboard_open:
                threading.Thread(target=show_virtual_keyboard, 
                args=(get_current_pinch_location,), daemon=True).start()
        active_app = name
        feedback(f"{app_name.capitalize()} Launched")
    except Exception as e:
        print(f"Error launching {app_name}: {e}")
def perform_app_action(app, action):
    try:
        app = str(app).lower()
        action = str(action).lower()
        if app == "chrome":
            if action == "new_tab": pyautogui.hotkey("ctrl","t")
            elif action == "scroll_down": pyautogui.scroll(-600)
            elif action == "refresh": pyautogui.hotkey("ctrl","r")
        elif app == "youtube":
            if action == "play_pause": pyautogui.press("space")
            elif action == "volume_up": pyautogui.press("volumeup")
            elif action == "next_video": pyautogui.hotkey("shift","n")
        elif app == "vscode":
            if action == "save_file": pyautogui.hotkey("ctrl","s")
            elif action == "run_code": pyautogui.hotkey("ctrl","f5")
            elif action == "open_terminal": pyautogui.hotkey("ctrl","`")
        feedback(f"{action.replace('_',' ').capitalize()} in {app}")
    except Exception as e:
        print("Error performing app action:", e)

# ---------------------- Gesture Detection ----------------------
TIP = {"index":8,"middle":12,"ring":16,"pinky":20,"thumb":4}
PIP = {"index":6,"middle":10,"ring":14,"pinky":18,"thumb":3}
def lm(hand, idx): return hand.landmark[idx]
def finger_extended(hand, finger): return lm(hand,TIP[finger]).y < lm(hand,PIP[finger]).y
def is_palm(hand): return all(finger_extended(hand,f) for f in ["index","middle","ring","pinky"])
def is_rock(hand): return finger_extended(hand,"index") and not finger_extended(hand,"middle") and not finger_extended(hand,"ring") and finger_extended(hand,"pinky")
def is_victory(hand): return finger_extended(hand,"index") and finger_extended(hand,"middle") and not finger_extended(hand,"ring") and not finger_extended(hand,"pinky")
def is_fist(hand): return not any(finger_extended(hand,f) for f in ["index","middle","ring","pinky"])
def is_ok(hand,ok_dist=THRESH["ok_distance"]):
    ix, iy = lm(hand,TIP["index"]).x, lm(hand,TIP["index"]).y
    tx, ty = lm(hand,TIP["thumb"]).x, lm(hand,TIP["thumb"]).y
    d = np.hypot(ix-tx, iy-ty)
    return d<ok_dist and all(finger_extended(hand,f) for f in ["middle","ring","pinky"])
def is_thumbs_up(hand):
    ty = lm(hand,TIP["thumb"]).y
    return all(ty < lm(hand,TIP[f]).y for f in ["index","middle","ring","pinky"])
def is_thumbs_down(hand):
    ty = lm(hand,TIP["thumb"]).y
    return all(ty > lm(hand,TIP[f]).y for f in ["index","middle","ring","pinky"])
GESTURE_CHECKS = {
    "PALM": is_palm,"ROCK": is_rock,"VICTORY": is_victory,
    "FIST": is_fist,"OK": is_ok,"THUMBSUP": is_thumbs_up,"THUMBSDOWN": is_thumbs_down
}
def check_gesture(hand, gesture_name):
    fn = GESTURE_CHECKS.get(gesture_name.upper())
    return fn(hand) if fn is not None else False

# ---------------------- ML Integration ----------------------
USE_ML = True
MODEL_PATH = "gesture_knn_model.pkl"
ML_CONFIDENCE = 0.6
ml_model = None
ml_input_features = None
def load_ml_model():
    global ml_model, ml_input_features
    if not USE_ML: return
    if os.path.exists(MODEL_PATH):
        try:
            ml_model = joblib.load(MODEL_PATH)
            ml_input_features = getattr(ml_model,"n_features_in_", None)
            print(f"[ML] Model loaded ({ml_input_features} features).")
        except Exception as e:
            print("[ML] Failed to load model:", e)
            ml_model = None
    else:
        print(f"[ML] Model file not found: {MODEL_PATH}")
def extract_landmark_features(hand):
    wrist = lm(hand,0)
    coords = []
    dists = [np.hypot(l.x-wrist.x,l.y-wrist.y) for l in hand.landmark]
    scale = max(dists) if max(dists) > 1e-6 else 1.0
    for l in hand.landmark:
        coords.extend([(l.x-wrist.x)/scale,(l.y-wrist.y)/scale,(l.z-wrist.z)/scale])
    return np.array(coords,dtype=np.float32)
def ml_predict(hand):
    if ml_model is None: return None,0.0
    feats = extract_landmark_features(hand)
    X = feats
    if ml_input_features and len(feats)!=ml_input_features:
        if len(feats)>ml_input_features:
            X = feats[:ml_input_features]
        else:
            X = np.pad(feats,(0,ml_input_features-len(feats)),'constant')
    X = X.reshape(1,-1)
    try:
        if hasattr(ml_model,"predict_proba"):
            probs = ml_model.predict_proba(X)
            idx = int(np.argmax(probs,axis=1)[0])
            conf = float(probs[0][idx])
            pred = ml_model.classes_[idx]
        else:
            pred = ml_model.predict(X)[0]
            conf = 1.0
        return str(pred),conf
    except Exception as e:
        print("[ML] Prediction error:", e)
        return None,0.0
load_ml_model()

# ---------------------- Idle Mode ----------------------
def enter_idle_mode():
    global idle_mode, click_down, pinch_coords, pinch_active
    if not idle_mode:
        idle_mode = True
        try: pyautogui.mouseUp()
        except Exception: pass
        click_down = False
        pinch_coords[0] = None
        pinch_active[0] = False
        feedback("Idle mode activated")

# ---------------------- Main Loop ----------------------
while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        enter_idle_mode()
    else:
        if idle_mode: 
            idle_mode = False
            feedback("Normal mode activated")
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            index = hand_landmarks.landmark[8]
            thumb_lm = hand_landmarks.landmark[4]
            raw_x = np.interp(index.x,[0,1],[0,screen_width])
            raw_y = np.interp(index.y,[0,1],[0,screen_height])
            if not _first_detection:
                smooth_x,smooth_y = raw_x,raw_y
                _first_detection = True
            smooth_x += (raw_x-smooth_x)*SMOOTHING_FACTOR
            smooth_y += (raw_y-smooth_y)*SMOOTHING_FACTOR
            if not idle_mode:
                try: pyautogui.moveTo(smooth_x,smooth_y)
                except Exception: pass
            cv2.circle(frame,(int(index.x*w),int(index.y*h)),10,(0,255,0),-1)
            pinch_dist = np.hypot(index.x-thumb_lm.x,index.y-thumb_lm.y)
            middle = hand_landmarks.landmark[12]
            right_click_dist = np.hypot(middle.x-thumb_lm.x,middle.y-thumb_lm.y)
            wrist = hand_landmarks.landmark[0]
            fist_dist = np.hypot(index.y-wrist.y,thumb_lm.y-wrist.y)
            current_time = time.time()

            # Toggle app mode
            if check_gesture(hand_landmarks,APP_MODE_GESTURE) and current_time-last_toggle_time>TOGGLE_COOLDOWN:
                app_mode = not app_mode
                last_toggle_time = current_time
                feedback("App mode activated" if app_mode else "Normal mode activated")

            # Keyboard gesture
            if check_gesture(hand_landmarks,KEYBOARD_GESTURE) and not keyboard_open:
                keyboard_open = True
                threading.Thread(target=show_virtual_keyboard,args=(get_current_pinch_location,),daemon=True).start()
                feedback("Keyboard launched")
            elif USE_ML and ml_model and not keyboard_open:
                pred_label,conf = ml_predict(hand_landmarks)
                if pred_label and pred_label.upper()==KEYBOARD_GESTURE and conf>=ML_CONFIDENCE:
                    keyboard_open = True
                    threading.Thread(target=show_virtual_keyboard,args=(get_current_pinch_location,),daemon=True).start()
                    feedback(f"Keyboard launched (ML: {pred_label} {conf:.2f})")

            # Normal mode interactions
            if not app_mode and not idle_mode:
                if pinch_dist<THRESH["pinch_left_click"] and current_time-last_pinch_time>PINCH_COOLDOWN:
                    pyautogui.click()
                    last_pinch_time=current_time
                    cv2.putText(frame,"Left Click",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    pinch_coords[0]=pyautogui.position()
                else:
                    if pinch_dist>=THRESH["pinch_left_click"]:
                        pinch_coords[0]=None
                        pinch_active[0]=False
                if right_click_dist<THRESH["right_click"]:
                    pyautogui.rightClick()
                    cv2.putText(frame,"Right Click",(50,140),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                scroll_y=int(middle.y*h)
                if prev_y!=0:
                    dy=scroll_y-prev_y
                    if abs(dy)>10:
                        pyautogui.scroll(-50 if dy>0 else 50)
                        cv2.putText(frame,"Scroll",(50,180),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                prev_y=scroll_y
                if fist_dist<THRESH["drag_fist"] and not click_down:
                    pyautogui.mouseDown()
                    click_down=True
                    cv2.putText(frame,"Drag Start",(50,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,200),2)
                elif fist_dist>=THRESH["drag_fist"] and click_down:
                    pyautogui.mouseUp()
                    click_down=False
                    cv2.putText(frame,"Drag End",(50,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,200),2)
            # App mode interactions
            if app_mode:
                for gesture,app in GLOBAL_GESTURES.items():
                    if check_gesture(hand_landmarks,gesture):
                        if current_time-last_launch_time>LAUNCH_COOLDOWN:
                            launch_app(app)
                            last_launch_time=current_time
                if active_app and active_app in APP_SPECIFIC:
                    for gesture,action in APP_SPECIFIC[active_app].items():
                        if check_gesture(hand_landmarks,gesture):
                            perform_app_action(active_app,action)

    # Overlay gesture log
    y0=30
    for line in gesture_log:
        cv2.putText(frame,line,(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        y0+=25
    cv2.imshow(" Gesture Interface",frame)
    if cv2.waitKey(1)&0xFF==27: break
cap.release()
cv2.destroyAllWindows()
