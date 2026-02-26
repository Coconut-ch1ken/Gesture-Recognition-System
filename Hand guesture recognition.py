# Import necessary libraries
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import uuid
import os
import time
import streamlit as st

# Set the Streamlit page configuration
st.set_page_config(page_title="Customize")

# Dictionary to store gesture-event mappings
events = dict()

# Title of the Streamlit app
st.title("WaveMomentum")

# Loop to configure gestures and their corresponding actions
for i in range(5):
    # Select box to choose the gesture
    option = st.selectbox(
        'What gesture would you like to configure',
        ('Thumb_Up', 'Thumb_Down', 'Open_Palm', 'Closed_Fist', 'Pointing_Up'), key="a"+str(i))
    # Select box to choose the action for the selected gesture
    option2 = st.selectbox('What would you like this gesture to do',
        ('left', 'right', 'up', 'down', 'left-click', 'right-click', 'middle-click', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'), key="b"+str(i))

    # Display the selected gesture and action
    st.text(option)
    st.text(option2)
    # Store the gesture-action pair in the events dictionary
    events[i] = {
        "event": option,
        "action": option2,
    }
    st.text('---')

# Slider to set the confidence level for gesture recognition
confidence_level = st.slider("confidence level", 0.1, 1.0, 0.5)
st.write("confidence level is ", confidence_level)

# Print the confidence level to the console
print(confidence_level)

# Initialize MediaPipe drawing and hand solutions
mp_drawing = mp.tasks.vision.drawing_utils
mp_hands_connections = mp.tasks.vision.HandLandmarksConnections

# Timestamp for frames
frameTimeStamp = 0

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Initialize variables for timing
lastTime = 0
lastTimeBack = 0

# Create an empty Streamlit frame
stframe = st.empty()

# Initialize `results` variable to store gesture recognition results
results = None

# Define `multi` as a scaling factor, adjust as needed
multi = 1.0

# Callback function to handle gesture recognition results
def result(result, image, ms):
    global lastTime
    global lastTimeBack
    global results  # Update results with the recognized gestures
    results = result
    # Iterate through recognized gestures
    for gestures in result.gestures:
        for event in events:
            # Print the recognized gesture and the corresponding action
            print("event", events[event]["event"], "action", events[event]["action"], gestures[0].category_name)
            # If the recognized gesture matches the configured gesture and enough time has passed since the last gesture
            if gestures[0].category_name == events[event]["event"] and time.time() - lastTime >= 0.5:
                # Perform the configured action
                if "click" in events[event]["action"]:
                    pyautogui.click(button=str(events[event]["action"]).replace("-click", ""))
                else:
                    pyautogui.press(events[event]["action"])
                    print("up")
                # Update the last time a gesture was recognized
                lastTime = time.time()

# Create a GestureRecognizer with the specified options
with mp.tasks.vision.GestureRecognizer.create_from_options(
    mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="gesture_recognizer.task"),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        min_hand_detection_confidence=confidence_level,
        min_tracking_confidence=confidence_level,
        result_callback=result
    )
) as recog:
    # While the video capture is open
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the camera

        # Convert the frame to a MediaPipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Increment the frame timestamp
        frameTimeStamp += 1

        # Perform asynchronous gesture recognition
        recog.recognize_async(mp_image, frameTimeStamp)

        # Display the frame in the Streamlit app
        stframe.image(frame, channels="BGR", use_column_width=True)
        
        # Check if results and multi_hand_landmarks are available
        if results and results.hand_landmarks:
            pyautogui.moveTo(results.multi_hand_landmarks[0].landmark[8].x * pyautogui.size().width * multi, 
                             results.multi_hand_landmarks[0].landmark[8].y * pyautogui.size().height * multi)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()