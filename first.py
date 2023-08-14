import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

model = load_model('C:/Users/Kritt/vs_code/trained_model.h5')

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            height, width, _ = frame.shape
            hand_pts = np.zeros((21, 2), dtype=np.int32)

            for i, landmark in enumerate(hand_landmarks.landmark): 
                hand_pts[i] = (int(landmark.x * width), int(landmark.y * height))

            hand_image = frame[min(hand_pts[:, 1]):max(hand_pts[:, 1]), min(hand_pts[:, 0]):max(hand_pts[:, 0])]
            
            # Resize the hand_image to match the model's input shape
            hand_image_resized = cv2.resize(hand_image, (200, 200))
            
            hand_image_gray = cv2.cvtColor(hand_image_resized, cv2.COLOR_BGR2GRAY)
            hand_image_norm = hand_image_gray / 255.0
            hand_image_input = np.expand_dims(hand_image_norm, axis=-1)
            hand_image_input = np.expand_dims(hand_image_input, axis=0)

            prediction = model.predict(hand_image_input)
            predicted_class = np.argmax(prediction[0])
            predicted_letter = letters[predicted_class]

            cv2.putText(frame, predicted_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            print("Detected letter:", predicted_letter)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
