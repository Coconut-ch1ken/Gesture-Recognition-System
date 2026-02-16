import cv2
import mediapipe as mp

import pyautogui

import uuid
import os
import time

import streamlit as st

st.set_page_config(page_title="Customize")

events = dict()

st.title("WaveMomentum")


for i in range(5):
    option = st.selectbox(
    'What gesture would you like to configure',
    ('Thumb_Up', 'Thumb_Down', 'Open_Palm', 'Closed_Fist', 'Pointing_Up'), key="a"+str(i))
    option2 = st.selectbox('What would you like this gesture to do',
    ('left', 'right', 'up', 'down', 'left-click', 'right-click', 'middle-click', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'), key="b"+str(i))

    st.text(option)
    st.text(option2)
    events[i] = {
        "event": option,
        "action": option2,
    }
    st.text('---')

confidence_level = st.slider("confidence level",0.1,1.0,0.5)
st.write("confidence level is ",confidence_level)

print(confidence_level)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frameTimeStamp = 0

cap = cv2.VideoCapture(0)

lastTime = 0
lastTimeBack = 0

stframe = st.empty()

def result(result, image, ms):
    global lastTime
    global lastTimeBack
    for gestures in result.gestures:
        for event in events:
            print("event", events[event]["event"], "action", events[event]["action"], gestures[0].category_name)
            if gestures[0].category_name == events[event]["event"] and time.time() - lastTime >= 0.5:
                if "click" in events[event]["action"]:
                    pyautogui.click(button=str(events[event]["action"]).replace("-click", ""))
                else:
                    pyautogui.press(events[event]["action"])
                    print("up")
                lastTime = time.time()




with mp.tasks.vision.GestureRecognizer.create_from_options(
    mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="gestures.task"),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        min_hand_detection_confidence=confidence_level,
        min_tracking_confidence=confidence_level,
        result_callback=result
    )
) as recog:
    while cap.isOpened():
        ret,frame = cap.read()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame)

        frameTimeStamp += 1

        recog.recognize_async(mp_image, frameTimeStamp)

        stframe.image(frame, channels="BGR", use_column_width=True)
        if results.multi_hand_landmarks:
            pyautogui.moveTo(results.multi_hand_landmarks[0].landmark[8].x * pyautogui.size().width * multi, results.multi_hand_landmarks[0].landmark[8].y * pyautogui.size().height * multi)
           

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows
