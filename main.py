import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize drawing utility
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Define finger landmark indices
finger_landmarks = {
    'Thumb': [1, 2, 3, 4],
    'Index': [5, 6, 7, 8],
    'Middle': [9, 10, 11, 12],
    'Ring': [13, 14, 15, 16],
    'Pinky': [17, 18, 19, 20]
}

# Define fixed vertical positions for finger labels
finger_label_positions = {'Left': {'x': 10, 'y': 50}, 'Right': {'x': 400, 'y': 50}}
spacing = 30  # Space between finger labels

# Gesture recognition rules
def recognize_gesture(fingers_up):
    if all(fingers_up[finger] for finger in fingers_up):
        return "Open Hand"
    elif not any(fingers_up[finger] for finger in fingers_up):
        return "Fist"
    elif fingers_up['Index'] and fingers_up['Middle'] and not fingers_up['Ring'] and not fingers_up['Pinky']:
        return "Peace"
    else:
        return "Unknown Gesture"

# Create a named window for resizing
cv2.namedWindow('Hand Gesture Recognition', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)  # 1 means flip horizontally
    
    # Convert the frame to RGB (as required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks if any are detected
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw bones (lines) between landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the hand label
            hand_label = handedness.classification[0].label

            # Determine which fingers are up
            fingers_up = {finger: False for finger in finger_landmarks}

            for finger, landmarks in finger_landmarks.items():
                tip_id = landmarks[-1]  # Tip landmark index
                base_id = landmarks[0]  # Base landmark index
                tip_y = hand_landmarks.landmark[tip_id].y
                base_y = hand_landmarks.landmark[base_id].y

                # If tip is above the base, finger is up
                if tip_y < base_y:
                    fingers_up[finger] = True
            
            # Recognize gesture
            gesture = recognize_gesture(fingers_up)
            
            # Display finger status
            finger_text = [f'{finger} Up' for finger, is_up in fingers_up.items() if is_up]
            
            # Calculate bounding box coordinates
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Update bounding box coordinates
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

                # Draw pink circles on each landmark
                cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)  # Pink color
            
            # Display the hand label above the hand
            hand_label_x = int((min_x + max_x) / 2)
            hand_label_y = int(min_y - 10)  # Place above the hand
            cv2.putText(frame, f'{hand_label} Hand', (hand_label_x, hand_label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display which fingers are up in a vertical list
            if hand_label == 'Left':
                base_x = finger_label_positions['Left']['x']
                base_y = finger_label_positions['Left']['y']
            else:
                base_x = finger_label_positions['Right']['x']
                base_y = finger_label_positions['Right']['y']
            
            for i, finger in enumerate(finger_text):
                cv2.putText(frame, finger, (base_x, base_y + i * spacing), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the gesture label below the finger labels in pink
            gesture_x = base_x
            gesture_y = base_y + len(finger_text) * spacing + 40
            cv2.putText(frame, gesture, (gesture_x, gesture_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Get the original frame size
    height, width = frame.shape[:2]
    
    # Get the window size
    screen_width = cv2.getWindowImageRect('Hand Gesture Recognition')[2]
    screen_height = cv2.getWindowImageRect('Hand Gesture Recognition')[3]
    
    # Calculate the aspect ratio of the frame
    aspect_ratio = width / height
    
    # Resize the frame while keeping the aspect ratio
    if screen_width / screen_height > aspect_ratio:
        new_height = screen_height
        new_width = int(screen_height * aspect_ratio)
    else:
        new_width = screen_width
        new_height = int(screen_width / aspect_ratio)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Display the resized frame with landmarks
    cv2.imshow('Hand Gesture Recognition', resized_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
