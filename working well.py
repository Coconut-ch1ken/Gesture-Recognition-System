# Import necessary libraries
import cv2  # OpenCV for video capturing and processing
import mediapipe as mp  # Mediapipe for hand gesture recognition
import pyautogui  # PyAutoGUI for controlling the mouse and keyboard
import uuid  # UUID for generating unique identifiers
import os  # OS for interacting with the operating system
import time  # Time for time-related operations
import streamlit as st  # Streamlit for creating web applications

# Set the title for the web page
st.set_page_config(page_title="WaveMomentum")

# Create an empty dictionary to store user-defined events
events = dict()

# Create the title for the web page
st.title("WaveMomentum")

# Create selection boxes for users to customize gesture actions
for i in range(5):
    # Create a selection box for choosing a gesture
    option = st.selectbox(
        'What gesture would you like to configure',
        ('Thumb_Up', 'Thumb_Down', 'Open_Palm', 'Closed_Fist', 'Pointing_Up'),
        key="a" + str(i)
    )
    # Create a selection box for choosing an action
    option2 = st.selectbox(
        'What would you like this gesture to do',
        ('left', 'right', 'up', 'down', 'left-click', 'right-click', 'middle-click',
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
         'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'),
        key="b" + str(i)
    )

    st.text(option)
    st.text(option2)
    
    # Store the user's selections in the events dictionary
    events[i] = {
        "event": option,
        "action": option2,
    }
    
    st.text('-----------------------------')

detection_confidence_level = st.slider("detection confidence level",0.1,1.0,0.75)
st.write("detection confidence level is ",detection_confidence_level)

tracking_confidence_level = st.slider("tracking confidence level",0.1,1.0,0.5)
st.write("tracking confidence level is ",tracking_confidence_level)

cooldown = st.slider("cooldown time",0.1,10.0,0.5)
st.write("cooldown time is ",cooldown)

# Initialize Mediapipe drawing utilities (new mp.tasks API)
mp_drawing = mp.tasks.vision.drawing_utils

frameTimeStamp = 0

# Initialize video capture from the default camera (camera index 0)
cap = cv2.VideoCapture(0)

lastTime = 0
lastTimeBack = 0

toggle = False

# Create an empty Streamlit frame
stframe = st.empty()

# Store latest hand landmarks from the gesture recognition callback
latest_hand_landmarks = None

# Define a callback function for gesture recognition results
def result(result, image, ms):
    global lastTime
    global lastTimeBack
    global latest_hand_landmarks
    
    # Store hand landmarks for cursor control in the main loop
    if result.hand_landmarks:
        latest_hand_landmarks = result.hand_landmarks
    else:
        latest_hand_landmarks = None
    
    # Loop through detected gestures
    for gestures in result.gestures:
        for event in events:
            print("event", events[event]["event"], "action", events[event]["action"], gestures[0].category_name)
            
            # Check if the detected gesture matches a user-defined event
            if gestures[0].category_name == events[event]["event"] and time.time() - lastTime >= 0.5:
                if "click" in events[event]["action"]:
                    pyautogui.click(button=str(events[event]["action"]).replace("-click", ""))
                else:
                    pyautogui.press(events[event]["action"])
                    print("up")
                lastTime = time.time()

# Create a gesture recognizer using Mediapipe (no separate Hands() needed;
# GestureRecognizer already provides hand_landmarks in its result)
recog = mp.tasks.vision.GestureRecognizer.create_from_options(
    mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="gestures_recognizer.task"),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        min_hand_detection_confidence=detection_confidence_level,
        min_tracking_confidence=tracking_confidence_level,
        result_callback=result
    )
)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.flip(frame, 1)

    multi = 1

    # Use hand landmarks from the gesture callback for cursor control
    if latest_hand_landmarks and toggle:
        landmark_8 = latest_hand_landmarks[0][8]
        pyautogui.moveTo(
            landmark_8.x * pyautogui.size().width * multi,
            landmark_8.y * pyautogui.size().height * multi
        )

    # Convert to mediapipe image and run async gesture recognition
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    frameTimeStamp += 1
    recog.recognize_async(mp_image, frameTimeStamp)

    stframe.image(image, channels="BGR", use_column_width=True)
    

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()