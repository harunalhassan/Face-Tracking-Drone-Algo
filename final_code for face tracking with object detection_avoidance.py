import cv2
import numpy as np
import pygame
from ultralytics import YOLO
from djitellopy import Tello
import time
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
import threading

# --- Initialization ---
def init():
    pygame.init()
    pygame.display.set_mode((480, 480))

def getKey(keyName):
    pygame.event.pump()
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, f'K_{keyName}')
    return keyInput[myKey]

def getUltrasonicDistance():
    try:
        distance = tello.get_distance_tof()
        return distance if distance > 0 else None
    except:
        return None

def calculate_focal_length(known_distance, real_width, width_in_frame):
    return (width_in_frame * known_distance) / real_width if width_in_frame > 0 else 600

def estimate_distance(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame if width_in_frame > 0 else None

# --- Live Plotting ---
def live_plotting():
    plt.ion()
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    line, = ax.plot([], [], 'lime', linewidth=2)
    ax.set_ylim(0, 200)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Frame Count', color='white')
    ax.set_ylabel('Downward Distance (cm)', color='white')
    ax.set_title('Live Downward Distance', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    global distance_history, time_history
    while True:
        if len(distance_history) > 1:
            line.set_data(time_history, distance_history)
            ax.set_xlim(max(0, time_history[0]), max(50, time_history[-1]))
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.05)

# --- Gesture Detection ---
def detectPalmGesture(img):
    global tracking_enabled, last_gesture_time
    gesture_cooldown = 1.5
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = [lm.y for lm in hand_landmarks.landmark]
            fingers_extended = [
                lm_list[mp_hands.HandLandmark.INDEX_FINGER_TIP] < lm_list[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                lm_list[mp_hands.HandLandmark.MIDDLE_FINGER_TIP] < lm_list[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                lm_list[mp_hands.HandLandmark.RING_FINGER_TIP] < lm_list[mp_hands.HandLandmark.RING_FINGER_PIP],
                lm_list[mp_hands.HandLandmark.PINKY_TIP] < lm_list[mp_hands.HandLandmark.PINKY_PIP]
            ]
            if current_time - last_gesture_time > gesture_cooldown:
                if all(fingers_extended):
                    tracking_enabled = True
                    last_gesture_time = current_time
                elif not any(fingers_extended):
                    tracking_enabled = False
                    last_gesture_time = current_time
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return img

# --- Face Detection & Tracking ---
def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    myFaceListC, myFaceListArea = [], []

    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        area = w * h
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if myFaceListArea:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    return img, [[0, 0], 0]

def trackFace(info, w, pid, pError):
    area = info[1]
    x = info[0][0]
    fb, yv = 0, 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -30, 30))

    if 6500 < area < 9000:
        fb = 0
    elif area > 9000:
        fb = -10
    elif area < 6500 and area > 5000:
        fb = 10

    if x == 0:
        yv = 0
        error = 0
    else:
        yv = speed

    tello.send_rc_control(0, fb, 0, yv)
    return error

# --- Keyboard Control ---
def getKeyboardInput(front_distance, down_distance):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if getKey("a"): lr = -speed
    elif getKey("d"): lr = speed
    if getKey("w") and (front_distance is None or front_distance > 50): fb = speed
    elif getKey("s"): fb = -speed
    if getKey("UP") and (down_distance is None or down_distance > 30): ud = speed
    elif getKey("DOWN") and (down_distance is None or down_distance > 10): ud = -speed
    if getKey("LEFT"): yv = -speed
    elif getKey("RIGHT"): yv = speed

    if getKey("t"): tello.takeoff(); time.sleep(2)
    if getKey("l"): tello.land(); time.sleep(2)

    return [lr, fb, ud, yv]

# --- Main Execution ---
init()
tello = Tello()
tello.connect()
tello.streamon()
print(f"Battery: {tello.get_battery()}%")

model = YOLO("yolov8n.pt")
KNOWN_DISTANCE = 50.0
KNOWN_WIDTH = 20.0

distance_history, time_history = deque(maxlen=50), deque(maxlen=50)
frame_counter = 0

plot_thread = threading.Thread(target=live_plotting, daemon=True)
plot_thread.start()

# Face tracking params
tracking_enabled = False
last_gesture_time = 0
pError = 0
pid = [0.4, 0.4, 0]
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

focal_length = 600
while True:
    frame = tello.get_frame_read().frame
    if frame is None:
        continue

    frame = cv2.resize(frame, (960, 720))  # Wider and clearer

    front_distance = 100
    results = model(frame)
    objects = results[0].boxes

    # Object Detection for Obstacle Avoidance
    for box in objects.xyxy:
        x1, y1, x2, y2 = map(int, box)
        box_width = x2 - x1
        front_distance = estimate_distance(focal_length, KNOWN_WIDTH, box_width)
        if front_distance:
            cv2.putText(frame, f"Front Distance: {front_distance:.2f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if front_distance < 20:
                cv2.putText(frame, "WARNING: TOO CLOSE!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                tello.send_rc_control(0, -30, 0, 0)
                time.sleep(0.1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    down_distance = getUltrasonicDistance()
    if down_distance:
        distance_history.append(down_distance + 30)
        time_history.append(frame_counter)
        frame_counter += 1
        if down_distance < 30:
            cv2.putText(frame, "WARNING: MOVE UP!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # tello.send_rc_control(0, 0, 30, 0)
            time.sleep(0.1)

    # Gesture Control & Face Tracking
    frame = detectPalmGesture(frame)
    if tracking_enabled:
        frame, info = findFace(frame)
        pError = trackFace(info, 480, pid, pError)
        cv2.putText(frame, "Tracking: ON", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking: OFF", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Manual Controls
    vals = getKeyboardInput(front_distance, down_distance)
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    cv2.putText(frame, f"Battery: {tello.get_battery()}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Tello Obstacle Avoidance + Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        tello.land()
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()
