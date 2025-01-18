import pickle
import cv2
import mediapipe as mp
import numpy as np

with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {i: chr(i + 65) for i in range(26)}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                x_.append(x)
                y_.append(y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), circle_radius=2)
            )

        x1 = int(min(x_) * W) - 10
        x2 = int(max(x_) * W) + 10
        y1 = int(min(y_) * H) - 10
        y2 = int(max(y_) * H) + 10

        if len(data_aux) == 42:
            pred = model.predict([np.asarray(data_aux)])
            pred_char = labels_dict[int(pred[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, pred_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Sign Detected', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
