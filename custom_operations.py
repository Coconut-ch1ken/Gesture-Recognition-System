import cv2
import mediapipe as mp

import pyautogui

import uuid
import os
import time

import streamlit as st

import numpy as np

st.set_page_config(page_title="Customize")

events = dict()

for i in range(5):

    st.title("WaveMomentum")
    option = st.selectbox(
    'What gesture would you like to configure',
    ('Thumbs_Up', 'Thumbs_Down', 'Open_Palm'), key="a"+str(i))
    option2 = st.selectbox('What would you like this gesture to do',
    ('left', 'right', 'up'), key="b"+str(i))

    st.text(option)
    st.text(option2)
    events[i] = {
        "event": option,
        "action": option2,
    }
    st.text(events)
    st.text('---')

confidence_level = st.slider("confidence level",0.0,1.0,0.5)
st.write("confidence level is ",confidence_level)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frameTimeStamp = 0

cap = cv2.VideoCapture(0)

lastTime = 0
lastTimeBack = 0

def result(result, image, ms):
    global lastTime
    global lastTimeBack
    for gestures in result.gestures:
        for event in events:
            print("event", events[event]["event"], "action", events[event]["action"])
            if gestures[0].category_name == events[event]["event"] and time.time() - lastTime >= 2:
                pyautogui.press(events[event]["action"])
                print("up")
                lastTime = time.time()




with mp.tasks.vision.GestureRecognizer.create_from_options(
    mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="gestures_recognizer.task"),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        min_hand_detection_confidence=1,
        min_tracking_confidence=1,
        result_callback=result
    )
) as recog:
    while cap.isOpened():
        ret,frame = cap.read()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame)

        frameTimeStamp += 1

        recog.recognize_async(mp_image, frameTimeStamp)

        cv2.imshow('Hand Tracking', frame)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows