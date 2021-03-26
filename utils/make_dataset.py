import cv2
import numpy as np
import csv
import time
from global_var import *


def update_count():
    with open(path_to_save, 'r') as csv_reader:
        csv_r = csv.reader(csv_reader)
        count_lines = len(list(csv_r))
    return count_lines


cap = cv2.VideoCapture(DEVICE_ID)

record = False
count = 0
frame_count = 0
gest = 'А'

path_to_save = f'../dataset/letters/{gest}.csv'

start = time.time()
with open(path_to_save, 'a+') as file:
    csv_file = csv.writer(file)
    count = update_count()
    while cap.isOpened():
        _, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hand.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            if record and frame_count % 2 == 0:
                if count >= 5000:
                    record = False
                    print('STOP RECORDING')
                    continue

                points = []
                for mark in hand_landmarks.landmark:
                    points.extend([mark.x, mark.y, mark.z])

                points.append(ALPH.index(gest))
                csv_file.writerow(points)
                count += 1
                if count % 5 == 0:
                    print(f'{count} recorded')

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_count += 1
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(5)
        if key & 0xFF == 27 or key == ord('q'):
            break
        elif key == ord('z'):
            if record:
                print('STOP RECORDING')
                record = False
            else:
                print('START RECORDING')
                record = True
        elif key == ord('u'):
            count = update_count()
            print('UPDATED')

print(f'ELAPSED TIME: {np.round((time.time() - start) / 60, 2)} m')
cap.release()

