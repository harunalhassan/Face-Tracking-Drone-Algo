import cv2
import numpy as np
import pygame
from djitellopy import Tello
import time
import mediapipe as mp

# Initialize Pygame for keyboard control
def init():
    pygame.init()
    pygame.display.set_mode((480, 480))

def getKey(keyName):
    pygame.event.pump()
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, f'K_{keyName}')
    return keyInput[myKey]

# Initialize Tello Drone
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()

# Face Tracking Variables
w, h = 480, 480
fbRange = [6500, 9000]
pid = [0.4, 0.4, 0]
pError = 0
takeoff = False
tracking_enabled = False

# MediaPipe setup for palm detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
last_gesture_time = 0  # Cooldown time tracker

# Gesture Recognition: Open Palm = Start, Fist = Stop
def detectPalmGesture(img):
    global tracking_enabled, last_gesture_time
    gesture_cooldown = 1.5  # seconds
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

# Face Detection
def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

# Face Tracking Logic
def trackFace(info, w, pid, pError):
    area = info[1]
    x, y = info[0]

    fb = 0
    yv = 0

    error = x - (w // 2)
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -30, 30))

    if fbRange[0] < area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -10
    elif 5000 < area < fbRange[0]:
        fb = 10

    if x == 0:
        yv = 0
        error = 0
    else:
        yv = speed

    tello.send_rc_control(0, fb, 0, yv)
    return error

# Manual Keyboard Control
def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    global takeoff

    if getKey("a"):
        lr = -speed
    elif getKey("d"):
        lr = speed

    if getKey("w"):
        fb = speed
    elif getKey("s"):
        fb = -speed

    if getKey("UP"):
        ud = speed
    elif getKey("DOWN"):
        ud = -speed

    if getKey("LEFT"):
        yv = -speed
    elif getKey("RIGHT"):
        yv = speed

    if getKey("t") and not takeoff:
        tello.takeoff()
        time.sleep(2)
        takeoff = True

    if getKey("l"):
        tello.land()
        time.sleep(2)
        takeoff = False

    return [lr, fb, ud, yv]

# Start
init()

while True:
    img = tello.get_frame_read().frame
    img = cv2.resize(img, (w, h))

    # Gesture Detection
    img = detectPalmGesture(img)

    # Face Tracking
    if tracking_enabled:
        img, info = findFace(img)
        pError = trackFace(info, w, pid, pError)
        cv2.putText(img, "Tracking: ON", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(img, "Tracking: OFF", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Manual Override
    vals = getKeyboardInput()
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    cv2.imshow("Tello Face Tracking with Palm Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()
