import cv2
import mediapipe as mp
import pyautogui
import time
import platform

# Initialize MediaPipe's Hand module (new mp.tasks API)
mp_drawing = mp.tasks.vision.drawing_utils
mp_hands_connections = mp.tasks.vision.HandLandmarksConnections

# Create a HandLandmarker for hand detection
hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="gestures_recognizer.task"),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
)

# Initialize variables for gesture action timing
lastTime = 0
action_interval = 0.1  # Minimum time interval between actions in seconds

# Define a function to perform actions based on the recognized gesture
def perform_action(gesture):
    global lastTime

    if time.time() - lastTime >= action_interval:
        if gesture == "Thumb_Up":
            if platform.system() == 'Darwin':
                pyautogui.hotkey('command', 's')
            else:
                pyautogui.hotkey('ctrl', 's')

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

    # Convert the frame to a mediapipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process the frame and detect hands
    results = hand_landmarker.detect(mp_image)

    # If hands are detected, draw landmarks and perform actions based on gestures
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands_connections.HAND_CONNECTIONS)
            
            # Get landmark positions
            thumb_tip = hand_landmarks[mp.tasks.vision.HandLandmarker.HandLandmarkIndex.THUMB_TIP] if hasattr(mp.tasks.vision, 'HandLandmarkIndex') else hand_landmarks[4]
            index_finger_tip = hand_landmarks[8]
            middle_finger_tip = hand_landmarks[12]

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