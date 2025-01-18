import os
import cv2

data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

number_of_classes = 26
class_images = 500

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(data_dir, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting images for class {j}. Press "c" to start.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot access webcam.")
            break

        cv2.putText(frame, 'Press "c" to start capturing', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):  
            break

    print(f'Starting to capture images for class {j}...')
    counter = 0

    while counter < class_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot access webcam.")
            break

        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)

        cv2.putText(frame, f'Class {j}, Image {counter + 1}/{class_images}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            print("Exiting data collection.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        counter += 1

    print(f'Completed capturing images for class {j}.')

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
