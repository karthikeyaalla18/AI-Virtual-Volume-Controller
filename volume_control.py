import cv2
import mediapipe as mp
import math
import numpy as np
import os

# --- 1. SETUP CAMERA & HAND TRACKER ---
wCam, hCam = 640, 480 
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Track previous volume to prevent "command spamming" (lag)
previous_vol = 0

print("✅ System Active. Press 'q' to exit.")

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip image so it acts like a mirror
    img = cv2.flip(img, 1)
    
    # Convert to RGB for MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    lmList = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

    # --- 2. GESTURE LOGIC ---
    if len(lmList) != 0:
        # Landmark 4 = Thumb Tip, Landmark 8 = Index Tip
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw visual markers
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Calculate distance
        length = math.hypot(x2 - x1, y2 - y1)

        # Map Hand Range (30px - 200px) to Volume (0 - 100)
        vol = np.interp(length, [30, 200], [0, 100])
        
        # --- 3. MAC SPECIFIC VOLUME CONTROL ---
        # Only run the command if volume changes by > 3% to prevent lag
        if abs(vol - previous_vol) > 3:
            # osascript is the Mac command to control system settings
            os.system(f"osascript -e 'set volume output volume {int(vol)}'")
            previous_vol = vol

        # Display Volume % on screen
        cv2.putText(img, f'Vol: {int(vol)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 3)

    cv2.imshow("Mac Gesture Control", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()