# HandGestureAI
 Hand Gesture Recognition with MediaPipe and OpenCV
This project is a hand gesture recognition system that uses MediaPipe, OpenCV, and Python to track hand movements and detect which fingers are raised. The program captures video from the webcam, processes the frames to detect hands, and labels which fingers are raised for both left and right hands.

Features:
>Hand Detection: Detects hands in real-time using MediaPipe's pre-trained model.
>Landmark Detection: Tracks 21 hand landmarks for each hand.
>Finger Detection: Identifies if the thumb, index, middle, ring, or pinky fingers are raised.
>Labels: Displays finger labels in a vertical list beside each hand. The hand (left or right) is labeled above the hand.

Dependencies:
Before running the code, ensure you have the following libraries installed:
>   pip install numpy opencv-python mediapipe
Additionally, if you don't have a package manager or virtual environment set up, you might need to install Pip or use a tool like Anaconda for environment management.

List of Dependencies:
>numpy: For numerical computations
>opencv-python: To capture video from the webcam and perform frame processing
>mediapipe: Pre-trained hand detection and landmark tracking model

Running the Code:
1.Clone or download this repository.

2.Install the required libraries by running the following command:
>   pip install numpy opencv-python mediapipe

3.Run the Python script using the following command:
>   python hand_gesture_recognition.py

4.The webcam will open, and the system will start detecting and labeling hands and fingers in real-time.
>   Hand Labels: The hand will be labeled as either "Left Hand" or "Right Hand" above the respective hand.
>   Finger Labels: When a finger is raised, it will be labeled in a vertical list beside the corresponding hand.

5.Press q to exit the program.

Code Explanation:
>   Hand Detection: Uses MediaPipe's Hands() model to detect hand landmarks.
>   Landmarks: The 21 landmarks for each hand are drawn, and their positions are used to calculate which fingers are raised.
>   Labeling: For each hand, the program labels the hand ("Left Hand" or "Right Hand") above the hand. Finger states (up or down)     are displayed in a list beside the hand.

Customization:
>   Resize Factor: You can adjust the resize_factor variable to change the size of the output window.
>   Finger Label Positioning: The vertical spacing between finger labels can be adjusted using the spacing variable, and the label positions can be customized in the finger_label_positions dictionary.

Troubleshooting:
>   If the webcam does not open, ensure that cv2.VideoCapture(0) is using the correct index for your webcam.
>   If the finger labels appear off-screen, adjust the finger_label_positions for both the left and right hands.

License:
This project is licensed under the MIT License.
