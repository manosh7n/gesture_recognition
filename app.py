import cv2
import pickle
import numpy as np
import mediapipe as mp
from scipy.stats import mode
from classifier.classifier import Classifier


def show_predict(predict, frame, pos, clf_name):
    if pos == 0:
        shift = frame.shape[1]-80
        shift_y = 38
        cv2.rectangle(frame, (shift-5, 22), (frame.shape[1]-3, 5), (255, 255, 255), thickness=-1)
        cv2.putText(frame, clf_name,
                    (shift, 20), cv2.FONT_HERSHEY_COMPLEX, 0.47,
                    (0, 0, 0), 2)
    elif pos == 1:
        shift = 5
        shift_y = 38
        cv2.rectangle(frame, (shift, 22), (75, 5), (255, 255, 255), thickness=-1)
        cv2.putText(frame, clf_name,
                    (shift, 20), cv2.FONT_HERSHEY_COMPLEX, 0.47,
                    (0, 0, 0), 2)
    else:
        shift = frame.shape[1] - 80
        shift_y = 235
        cv2.rectangle(frame, (shift - 5, 219), (frame.shape[1] - 3, 205), (255, 255, 255), thickness=-1)
        cv2.putText(frame, clf_name,
                    (shift, 217), cv2.FONT_HERSHEY_COMPLEX, 0.47,
                    (0, 0, 0), 2)
    pred_sign = np.argmax(predict)
    for i, sign in enumerate(ALPH):
        cv2.putText(frame, f'{sign}: {predict[i]:.2f}',
                    (shift, shift_y + (20*i+1)), cv2.FONT_HERSHEY_COMPLEX, 0.62,
                    (50, 224, 30) if pred_sign == i else (20, 20, 230), 2)


def draw_crop_bb(frame, x, y):
    # Find min and max (x, y) to draw bounding box
    max_y, min_y = int(np.max(y) * 480), int(np.min(y) * 480)
    max_x, min_x = int(np.max(x) * 640), int(np.min(x) * 640)
    # Increase the diagonal(in pixels) for a bounded box
    diag_incr = 75
    # 4 coordinates (x, y) of bb
    bbox_pts = np.array([[min_x - diag_incr, min_y - diag_incr],
                         [max_x + diag_incr, min_y - diag_incr],
                         [min_x - diag_incr, max_y + diag_incr],
                         [max_x + diag_incr, max_y + diag_incr]])
    x, y, w, h = np.abs(cv2.boundingRect(np.reshape(bbox_pts, (-1, 4, 2))))
    crop = frame[y:y + h, x:x + w].copy()
    # Draw bounding box
    cv2.rectangle(frame, (max_x + diag_incr, min_y - diag_incr),
                  (min_x - diag_incr, max_y + diag_incr), (0, 32, 222), 2)
    # Show crop hand in another window
    cv2.imshow('Palm', cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR))
    cv2.moveWindow('Palm', 20, 20)
    return crop


def draw_progress_bar(frame, frame_count, word):
    cv2.rectangle(frame, (5, 463), (630, 477), (233, 111, 0), -1)
    cv2.rectangle(frame, (5, 463), (int(630 * frame_count / 30), 477), (111, 233, 0), -1)
    cv2.rectangle(frame, (5, 440), (630, 460), (250, 250, 250), -1)
    cv2.putText(frame, word, (5, 458), cv2.FONT_HERSHEY_COMPLEX, 0.7, (20, 20, 20), 2)


WINDOW = "Gesture recognition"
CLASSIFIER_MODEL_PATH = "models/model_mobile9.h5"
KEY_POINTS_CLASSIFIER_PATH = 'models/lr_model.sav'
DEVICE_ID = 0
ALPH = ['А', 'Б', 'В', 'Г', 'И', 'К', 'Н', 'О', 'С']

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('videos/output_01.avi', fourcc, 10, (640, 480), True)
classifier = Classifier(CLASSIFIER_MODEL_PATH)
loaded_model = pickle.load(open(KEY_POINTS_CLASSIFIER_PATH, 'rb'))
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(DEVICE_ID)
cv2.namedWindow(WINDOW)

frame_count = 0
predictions = []
word = ''

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        frame_count += 1
        x_coo = []
        y_coo = []
        pts_to_pred = []

        for hand_landmarks in results.multi_hand_landmarks:
            for mark in hand_landmarks.landmark:
                pts_to_pred.extend([mark.x, mark.y, mark.z])
                x_coo.append(mark.x)
                y_coo.append(mark.y)

            crop = draw_crop_bb(image, x_coo, y_coo)
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            pts_to_pred = np.array(pts_to_pred).reshape(1, -1)
            pred_pts = loaded_model.predict_proba(pts_to_pred)[0]
            pred_cnn = classifier(crop)[0]
            mean_pred = (pred_cnn*0.4 + pred_pts*1.6) / 2
            predictions.append(np.argmax(mean_pred))

            # show_predict(pred_pts, image, 0, '  Points')
            # show_predict(pred_cnn, image, 1, ' CNN')
            show_predict(mean_pred, image, 1, '  Predict')

    draw_progress_bar(image, frame_count, word)
    if frame_count == 30:
        mode_predict, val_count = mode(predictions)
        if val_count > frame_count // 2:
            word += ALPH[mode_predict[0]]
        frame_count = 0
        predictions = []
    if writer is not None:
        writer.write(image)

    cv2.imshow(WINDOW, image)
    key = cv2.waitKey(5)
    if key & 0xFF == 27:
        break
    if key == ord('c') or len(word) >= 45:
        word = ''
        frame_count = 0

if writer is not None:
    writer.release()

hands.close()
cap.release()
