import cv2
import mediapipe as mp
import csv


def update_count():
    with open(path_to_save, 'r') as csv_reader:
        csv_r = csv.reader(csv_reader)
        count_lines = len(list(csv_r))
    return count_lines

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
hand = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

record = False
count = 0
frame_count = 0
gest = 'А'
alph = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К',
        'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц',
        'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
path_to_save = f'../dataset/letters/{gest}.csv'

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
                points = []
                for mark in hand_landmarks.landmark:
                    points.extend([mark.x, mark.y, mark.z])

                points.append(alph.index(gest))
                csv_file.writerow(points)
                count += 1
                if count % 5 == 0:
                    print(f'{count} recorded')

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_count += 1
        cv2.imshow('Make dataset', image)
        key = cv2.waitKey(5)
        if key & 0xFF == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            if record:
                print('STOP RECORDING')
                record = False
            else:
                print('START RECORDING')
                record = True
        elif key == ord('u'):
            count = update_count()
cap.release()
