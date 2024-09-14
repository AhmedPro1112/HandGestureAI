import numpy as np
print(np.__version__)
import pandas as pd
import cv2
import tensorflow as tf
from keras.models import Sequential
import matplotlib.pyplot as plt
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize drawing utility
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Convert the frame to RGB (as required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks if any are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw bones (lines) between landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Loop through all 21 landmarks
            for landmark in hand_landmarks.landmark:
                # Get the x and y coordinates of each landmark
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Draw pink circles on each landmark
                cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)  # Pink color

    
    # Display the frame with landmarks
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
