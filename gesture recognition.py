import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe's Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for gesture action timing
lastTime = 0
action_interval = 0.1  # Minimum time interval between actions in seconds

# Define a function to perform actions based on the recognized gesture
def perform_action(gesture):
    global lastTime

    if time.time() - lastTime >= action_interval:
        if gesture == "Thumb_Up":
            pyautogui.press(['ctrl', 's'])

        elif gesture == "Thumb_Down":
            pyautogui.press('backspace')
        
        elif gesture == "Pointing_Up":
            pyautogui.press('up')

        elif gesture == "Pointing_Down":
            pyautogui.press('down')

        # elif gesture == "Open_Palm":
        #     pyautogui.press('space')
        # elif gesture == "Closed_Fist":
        #     pyautogui.press('enter')
        lastTime = time.time()

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    # If hands are detected, draw landmarks and perform actions based on gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Placeholder: Add your own logic to recognize gestures based on landmarks
            # Here we'll just print the landmarks for simplicity
            # print(hand_landmarks)

            # Example logic to recognize gestures (this is very simplistic)
            # Add your own complex gesture recognition logic here
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Simple logic to recognize some gestures (for demonstration purposes)
            if thumb_tip.y < index_finger_tip.y and thumb_tip.y < middle_finger_tip.y:
                gesture = "Thumb_Up"
            elif thumb_tip.y > index_finger_tip.y and thumb_tip.y > middle_finger_tip.y:
                gesture = "Thumb_Down"
            elif index_finger_tip.y < middle_finger_tip.y:
                gesture = "Pointing_Up"
            elif index_finger_tip.y > middle_finger_tip.y:
                gesture = "Pointing_Down"
            else:
                gesture = "Open_Palm"
            
            # Perform action based on the recognized gesture
            perform_action(gesture)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()