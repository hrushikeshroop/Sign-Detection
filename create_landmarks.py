import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

data_dir = './data'
data = []
labels = []

for dir_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, dir_name)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        img_full_path = os.path.join(class_dir, img_name)
        if not img_full_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Unable to read image {img_full_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                auxdata = []
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, y_min = min(x_coords), min(y_coords)
                for lm in hand_landmarks.landmark:
                    auxdata.append(lm.x - x_min)
                    auxdata.append(lm.y - y_min)
                data.append(auxdata)
                labels.append(int(dir_name))

output_file = 'data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data collection and saving complete. Processed data saved to {output_file}.")
