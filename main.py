import cv2
import mediapipe as mp
import os
from collections import deque

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Configuration
finger_landmarks = {
    'Thumb': [1, 2, 3, 4],
    'Index': [5, 6, 7, 8],
    'Middle': [9, 10, 11, 12],
    'Ring': [13, 14, 15, 16],
    'Pinky': [17, 18, 19, 20]
}

motion_history = {
    'Left': deque(maxlen=15),
    'Right': deque(maxlen=15)
}

DISPLAY_CONFIG = {
    'positions': {'Left': (10, 50), 'Right': (400, 50)},
    'spacing': 30,
    'colors': {'static': (255, 0, 255), 'motion': (0, 255, 255)}
}

GESTURE_PARAMS = {
    'MOTION_THRESHOLD': 40,
    'CIRCLE_THRESHOLD': 100
}

def recognize_static_gesture(fingers_up):
    try:
        # Basic gestures
        if all(fingers_up.values()):
            return "Hello"
        if not any(fingers_up.values()):
            return "Fist"
        if fingers_up['Index'] and fingers_up['Middle'] and not others_up(fingers_up, ['Index', 'Middle']):
            return "Peace"
        if fingers_up['Thumb'] and not others_up(fingers_up, ['Thumb']):
            return "Yes"
        if fingers_up['Index'] and not others_up(fingers_up, ['Index']):
            return "No"
        
        # Advanced gestures
        if (fingers_up['Thumb'] and fingers_up['Index'] and fingers_up['Pinky'] and 
            not others_up(fingers_up, ['Thumb', 'Index', 'Pinky'])):
            return "I Love You"
        if fingers_up['Thumb'] and fingers_up['Index'] and not others_up(fingers_up, ['Thumb', 'Index']):
            return "Okay"
        if (fingers_up['Middle'] and fingers_up['Ring'] and fingers_up['Pinky'] and 
            not others_up(fingers_up, ['Middle', 'Ring', 'Pinky'])):
            return "Metal"
        if fingers_up['Thumb'] and fingers_up['Pinky'] and not others_up(fingers_up, ['Thumb', 'Pinky']):
            return "Call Me"
        if (fingers_up['Index'] and fingers_up['Thumb'] and fingers_up['Middle'] and 
            not others_up(fingers_up, ['Index', 'Thumb', 'Middle'])):
            return "Three"
        
        return "Unknown"
    except KeyError as e:
        print(f"Gesture recognition error: {str(e)}")
        return "Unknown"

def others_up(fingers_up, exclude):
    return any(v for k, v in fingers_up.items() if k not in exclude)

def recognize_motion_gesture(positions):
    try:
        if len(positions) < 5:
            return None

        # Calculate movement vectors
        dx = []
        dy = []
        for i in range(1, len(positions)):
            dx.append(positions[i][0] - positions[i-1][0])
            dy.append(positions[i][1] - positions[i-1][1])

        if not dx or not dy:
            return None

        avg_dx = sum(dx) / len(dx)
        avg_dy = sum(dy) / len(dy)
        total_movement = sum(abs(x) + abs(y) for x, y in zip(dx, dy))

        if total_movement < GESTURE_PARAMS['MOTION_THRESHOLD']:
            return None

        # Direction analysis
        x_directions = [1 if x > 0 else -1 for x in dx if x != 0]
        direction_changes = sum(x_directions[i] != x_directions[i+1] 
                              for i in range(len(x_directions)-1)) if len(x_directions) > 1 else 0

        # Gesture patterns
        if direction_changes >= 2 and sum(abs(x) for x in dx) > GESTURE_PARAMS['CIRCLE_THRESHOLD']:
            return "Waving"
        
        if (max(p[0] for p in positions) - min(p[0] for p in positions) > GESTURE_PARAMS['CIRCLE_THRESHOLD'] and
            max(p[1] for p in positions) - min(p[1] for p in positions) > GESTURE_PARAMS['CIRCLE_THRESHOLD']):
            return "Circle (Wait)"

        # Directional movements
        if abs(avg_dy) > abs(avg_dx) * 2:
            return "Move Up" if avg_dy < 0 else "Move Down"
        
        if abs(avg_dx) > abs(avg_dy) * 2:
            return "Move Right" if avg_dx > 0 else "Move Left"
        
        return None
    except Exception as e:
        print(f"Motion detection error: {str(e)}")
        return None

# Main processing loop
cv2.namedWindow('Hand Gesture AI', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label

            # Finger state detection
            fingers_up = {finger: False for finger in finger_landmarks}
            for finger, landmarks in finger_landmarks.items():
                tip = hand_landmarks.landmark[landmarks[-1]]
                base = hand_landmarks.landmark[landmarks[0]]
                fingers_up[finger] = tip.y < base.y

            # Gesture recognition
            static_gesture = recognize_static_gesture(fingers_up)
            
            # Motion tracking
            wrist_pos = (hand_landmarks.landmark[0].x * frame.shape[1],
                         hand_landmarks.landmark[0].y * frame.shape[0])
            motion_history[hand_label].append(wrist_pos)
            motion_gesture = recognize_motion_gesture(list(motion_history[hand_label]))

            # Display information
            x, y = DISPLAY_CONFIG['positions']['Left' if hand_label == 'Left' else 'Right']
            finger_text = [f"{finger} Up" for finger, up in fingers_up.items() if up]

            # Draw static gesture
            cv2.putText(frame, f"Static: {static_gesture}",
                       (x, y + len(finger_text) * DISPLAY_CONFIG['spacing'] + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, DISPLAY_CONFIG['colors']['static'], 2)

            # Draw motion gesture
            if motion_gesture:
                cv2.putText(frame, f"Motion: {motion_gesture}",
                           (x, y + len(finger_text) * DISPLAY_CONFIG['spacing'] + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, DISPLAY_CONFIG['colors']['motion'], 2)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display window
    cv2.imshow('Hand Gesture AI', cv2.resize(frame, (800, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()