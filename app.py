import cv2
import pickle
import numpy as np
from utils.global_var import *


def show_predict(predict, frame):
    shift_x = 5
    shift_y = 38
    cv2.rectangle(frame, (shift_x, shift_y - 15), (75, shift_y - 30),
                  (255, 255, 255), thickness=-1)

    cv2.putText(frame, 'Алфавит', (shift_x, shift_y - 18),
                cv2.FONT_HERSHEY_COMPLEX, 0.47, (0, 0, 0), 2)
    pred_sign = np.argmax(predict)
    for i, sign in enumerate(classes):
        cv2.putText(frame, f'{sign}: {predict[i]:.2f}',
                    (shift_x, shift_y + (20 * i + 1)), cv2.FONT_HERSHEY_COMPLEX, 0.62,
                    (50, 224, 30) if pred_sign == i else (20, 20, 230), 2)


cap = cv2.VideoCapture(DEVICE_ID)
cv2.namedWindow(WINDOW)
# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter('videos/output_01.avi', fourcc, 10, (640, 480), True)

clf = pickle.load(open(KEY_POINTS_CLASSIFIER_PATH, 'rb'))

frame_count = 0
predictions = []

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hand.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # If there is a hand in the frame
    if results.multi_hand_landmarks:
        frame_count += 1
        points_xyz = []
        hand_landmarks = results.multi_hand_landmarks[0]

        for mark in hand_landmarks.landmark:
            points_xyz.extend([mark.x, mark.y, mark.z])

        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        points_xyz = np.array(points_xyz).reshape(1, -1)
        pred = clf.predict_proba(points_xyz)[0]

        show_predict(pred, image)

    cv2.imshow(WINDOW, image)

    # if writer is not None:
    #     writer.write(image)

    key = cv2.waitKey(5)
    if key & 0xFF == 27 or key == ord('q'):
        break


# if writer is not None:
#     writer.release()

hand.close()
cap.release()
