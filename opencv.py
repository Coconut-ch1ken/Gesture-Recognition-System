import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.tasks.vision.drawing_utils
mp_hands_connections = mp.tasks.vision.HandLandmarksConnections

cap = cv2.VideoCapture(0)

# Create a HandLandmarker for hand detection
hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="gestures_recognizer.task"),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.8,
        min_tracking_confidence=0.5
    )
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip on horizontal
    image = cv2.flip(frame, 1)
    
    # Convert to mediapipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Detect hands
    results = hand_landmarker.detect(mp_image)
    
    # Print results
    print(results)
    
    # Rendering results
    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand, mp_hands_connections.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )
    
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# mp_drawing.DrawingSpec??


os.makedirs('Output Images', exist_ok=True)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip on horizontal
    image = cv2.flip(frame, 1)
    
    # Convert to mediapipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Detect hands
    results = hand_landmarker.detect(mp_image)
    
    # Print results
    print(results)
    
    # Rendering results
    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand, mp_hands_connections.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )
    
    # Save our image    
    cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()