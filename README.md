# HandGestureAI

### Hand Gesture Recognition with MediaPipe and OpenCV

This project is a hand gesture recognition system that uses MediaPipe, OpenCV, and Python to track hand movements and detect which fingers are raised. The program captures video from the webcam, processes the frames to detect hands, and labels which fingers are raised for both left and right hands.

## Features:

- **Hand Detection**: Detects hands in real-time using MediaPipe's pre-trained model.
- **Landmark Detection**: Tracks 21 hand landmarks for each hand.
- **Finger Detection**: Identifies if the thumb, index, middle, ring, or pinky fingers are raised.
- **Labels**: Displays finger labels in a vertical list beside each hand. The hand (left or right) is labeled above the hand.

---

## Setup Instructions:

### 1. Install Python

If you don't have Python installed, follow the steps below:

#### **macOS/Linux:**

- Install Python using Homebrew (or download from [python.org](https://www.python.org)):
  ```bash
  brew install python
  ```

#### **Windows:**

- Download and install Python from the official site: [Download Python](https://www.python.org/downloads/).

Make sure `python3` and `pip3` are available in your terminal by running:

```bash
python3 --version
pip3 --version
```

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/HandGestureAI.git
cd HandGestureAI
```

---

## 3. Set Up Virtual Environment and Install Dependencies

It’s recommended to create a virtual environment to manage your dependencies.

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the Virtual Environment**:

   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```

3. **Install the Required Libraries**:

   Run the following command to install all required dependencies from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install the dependencies manually with specific versions:

   ```bash
   pip install numpy==1.26.4 opencv-python==4.11.0 mediapipe==0.10.21
   ```

   The project uses the following library versions:
   - OpenCV: 4.11.0
   - MediaPipe: 0.10.21
   - NumPy: 1.26.4

---

## Running the Code:

Once the environment is set up and the dependencies are installed, follow these steps to run the code:

1. **Run the Python script** using the following command:

   ```bash
   python main.py
   ```

2. **Functionality**:
   - The webcam will open, and the system will start detecting and labeling hands and fingers in real-time.
   - **Hand Labels**: The hand will be labeled as either "Left Hand" or "Right Hand" above the respective hand.
   - **Finger Labels**: When a finger is raised, it will be labeled in a vertical list beside the corresponding hand.
   
3. **Exit the Program**:
   - Press `q` to exit the program.

---

## Code Explanation:

- **Hand Detection**: Uses MediaPipe's `Hands()` model to detect hand landmarks.
- **Landmarks**: The 21 landmarks for each hand are drawn, and their positions are used to calculate which fingers are raised.
- **Labeling**: For each hand, the program labels the hand ("Left Hand" or "Right Hand") above the hand. Finger states (up or down) are displayed in a list beside the hand.

---

## Customization:

- **Resize Factor**: You can adjust the resize factor to change the size of the output window.
- **Finger Label Positioning**: The vertical spacing between finger labels can be adjusted using the `spacing` variable, and the label positions can be customized in the `finger_label_positions` dictionary.

---

## Troubleshooting:

- **Webcam Not Opening**: If the webcam does not open, ensure that `cv2.VideoCapture(0)` is using the correct index for your webcam.
- **Finger Labels Off-Screen**: If the finger labels appear off-screen, adjust the `finger_label_positions` for both the left and right hands.

---

## License:

This project is licensed under the MIT License.

---

This README now includes installation instructions for Python and the necessary libraries, along with clear steps for setting up the project and running the code. Let me know if you'd like to add more details or modifications!