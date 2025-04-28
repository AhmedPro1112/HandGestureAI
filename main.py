import cv2
import mediapipe as mp
import math
import random
import time

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
face_mesh = mp_face.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

# System states
PLAYING_GAME = False
LAST_COMPUTER_CHOICE_TIME = 0
COMPUTER_CHOICE = None
PLAYER_CHOICE = None
FACE_BOX_VISIBLE = False

def get_face_bounding_box(face_landmarks, frame):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in face_landmarks.landmark]
    ys = [lm.y * h for lm in face_landmarks.landmark]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

def detect_mood(face_landmarks):
    try:
        # Get key facial points
        mouth_top = face_landmarks.landmark[13]     # Upper lip center
        mouth_bottom = face_landmarks.landmark[14]  # Lower lip center
        mouth_left = face_landmarks.landmark[61]    # Left mouth corner
        mouth_right = face_landmarks.landmark[291]  # Right mouth corner
        mouth_center = face_landmarks.landmark[0]   # Chin center

        # Calculate mouth features
        mouth_openness = abs(mouth_top.y - mouth_bottom.y)
        # Print mouth_openness to the console
        print(f"Mouth Openness: {mouth_openness}, mouth corners: {mouth_left.y}, {mouth_right.y}")
        mouth_corners_avg = (mouth_left.y + mouth_right.y) / 2
        
        # Detect frown (sadness indicator)
        is_frowning = mouth_corners_avg > mouth_center.y + 0.02
        
        # Detect smile (happiness indicator)
        is_smiling = mouth_openness > 0.04
        

        # Mood determination
        if is_smiling:
            return "HAPPY 😊"
        elif is_frowning and mouth_openness < 0.05:
            return "SAD 😢"
        return "NEUTRAL 😐"
        
    except Exception as e:
        print(f"Mood detection error: {e}")
        return "NEUTRAL 😐"

def get_math_operation(hands_data):
    if len(hands_data) == 2:
        h1, h2 = hands_data
        if sum(h1['fingers']) >=4 and sum(h2['fingers']) >=4:
            return " = "
        if h1['fingers'][1] and h2['fingers'][1]:
            return " + "
    elif len(hands_data) == 1:
        hand = hands_data[0]
        if hand['fingers'][1] and not any(hand['fingers'][2:]):
            return " - "
    return None

def get_number(hands_data):
    total = 0
    for hand in hands_data:
        # Only count if hand is clearly showing numbers (all fingers down except counting ones)
        if sum(hand['fingers']) in {0,1,2,3,4,5}:
            total += sum(hand['fingers'][1:]) + (1 if hand['fingers'][0] else 0)
    return total if 0 <= total <= 10 else None

def handle_game(frame, hands_data):
    global COMPUTER_CHOICE, PLAYER_CHOICE, LAST_COMPUTER_CHOICE_TIME
    
    # Update computer choice every 3 seconds
    if time.time() - LAST_COMPUTER_CHOICE_TIME > 3:
        COMPUTER_CHOICE = random.choice(['ROCK', 'PAPER', 'SCISSORS'])
        LAST_COMPUTER_CHOICE_TIME = time.time()
    
    # Detect player choice
    current_choice = None
    if hands_data:
        fingers = sum(hands_data[0]['fingers'])
        current_choice = {
            0: 'ROCK',
            2: 'SCISSORS',
            5: 'PAPER'
        }.get(fingers, None)
    
    # Only update when valid choice detected
    if current_choice and current_choice != PLAYER_CHOICE:
        PLAYER_CHOICE = current_choice
    
    # Display game info
    if COMPUTER_CHOICE and PLAYER_CHOICE:
        results = {
            ('ROCK', 'SCISSORS'): "YOU WIN! 🎉",
            ('PAPER', 'ROCK'): "YOU WIN! 🎉", 
            ('SCISSORS', 'PAPER'): "YOU WIN! 🎉",
            ('ROCK', 'PAPER'): "YOU LOSE 😢",
            ('PAPER', 'SCISSORS'): "YOU LOSE 😢",
            ('SCISSORS', 'ROCK'): "YOU LOSE 😢"
        }
        result = results.get((PLAYER_CHOICE, COMPUTER_CHOICE), "DRAW! 🤝")
        
        y_pos = 50
        cv2.putText(frame, f"Player: {PLAYER_CHOICE}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Computer: {COMPUTER_CHOICE}", (20, y_pos+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, result, (20, y_pos+80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    return frame

def main():
    global PLAYING_GAME, COMPUTER_CHOICE, PLAYER_CHOICE, FACE_BOX_VISIBLE
    
    cap = cv2.VideoCapture(0)
    last_frame_time = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Limit to 15 FPS for stability
        if time.time() - last_frame_time < 0.066:
            continue
        last_frame_time = time.time()
        
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # Always process face for mood
        face_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mood = "NEUTRAL 😐"
        FACE_BOX_VISIBLE = False
        
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            mood = detect_mood(face_landmarks)
            
            # Draw face bounding box
            x_min, y_min, x_max, y_max = get_face_bounding_box(face_landmarks, frame)
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            FACE_BOX_VISIBLE = True
        
        # Always show mood top-left
        cv2.putText(display_frame, f"Mood: {mood}", (20, 30 if FACE_BOX_VISIBLE else 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Process hands
        hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands_data = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Simplified finger state detection
                fingers = [
                    hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x,  # Thumb
                    hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,   # Index
                    hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y, # Middle
                    hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y, # Ring
                    hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y  # Pinky
                ]
                hands_data.append({'fingers': fingers})

        # Game invitation system
        if not PLAYING_GAME:
            if "SAD" in mood:
                cv2.putText(display_frame, "Show thumbs up to play a game!", 
                           (20, display_frame.shape[0]-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                if hands_data and hands_data[0]['fingers'][0]:
                    PLAYING_GAME = True
                    COMPUTER_CHOICE = random.choice(['ROCK', 'PAPER', 'SCISSORS'])
            else:
                # Education mode
                math_op = get_math_operation(hands_data)
                number = get_number(hands_data)
                if math_op:
                    cv2.putText(display_frame, math_op, 
                               (display_frame.shape[1]//2-20, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif number is not None:
                    cv2.putText(display_frame, str(number), 
                               (display_frame.shape[1]//2-20, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Game mode
            display_frame = handle_game(display_frame, hands_data)
            if "HAPPY" in mood:
                cv2.putText(display_frame, "Smile detected! Exiting game...", 
                           (20, display_frame.shape[0]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 1)
                PLAYING_GAME = False
                COMPUTER_CHOICE = None

        cv2.imshow('AI Learning Companion', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()