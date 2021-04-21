import cv2
import pickle
import numpy as np
from scipy.stats import mode
from utils.Distance import *
from utils.GlobalVar import *
from more_itertools import unique_justseen as uj


cap = cv2.VideoCapture(DEVICE_ID)
cv2.namedWindow(WINDOW)
clf = pickle.load(open(KEY_POINTS_CLASSIFIER_PATH, 'rb'))
# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter('videos/output_01.avi', fourcc, 10, (1280, 720), True)

frame_count = 0
predictions = []
prev_predictions = []
isRecording = False
prev_length = 0


def show_predict(predict: np.ndarray, frame: np.ndarray, corner_coo: tuple[int, int]):
    """
    Displaying the classified gesture in the upper-right corner of the bounding box
    @param predict: array of predicted probabilities
    @param frame: the frame where the letter is displayed
    @param corner_coo: the coordinate where the letter is displayed
    """
    cv2.rectangle(frame, corner_coo, (corner_coo[0] + 70, corner_coo[1] - 70),
                  (255, 255, 255), thickness=-1)
    cv2.putText(frame, f'{classes[np.argmax(predict)]}', (corner_coo[0] + 18, corner_coo[1] - 18),
                cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 2)


def draw_bb(points: np.ndarray) -> (float, tuple[int, int]):
    """
        Draw bounding box and and returns the difference between the lengths of the diagonals between the frames
        (it is necessary to check whether the gesture changes or the same one is displayed), the coordinates
        of the upper-right corner of the bounding box for showing the prediction
    @param points: (x, y) coordinates or landmarks
    @return:
           percent_diff: percentage difference between the diagonal from the last frame and the current one
           right_up_corner: coordinate of right up corner
    """
    global prev_length
    x_max, x_min = int(np.max(points[0][::2]) * image.shape[1]), int(np.min(points[0][::2]) * image.shape[1])
    y_max, y_min = int(np.max(points[0][1::2]) * image.shape[0]), int(np.min(points[0][1::2]) * image.shape[0])
    right_up_corner = (x_max, y_min)
    line_length = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    diff = np.abs(line_length - prev_length)
    prev_length = line_length
    percent_diff = diff * 100 / line_length

    cv2.line(image, (x_max, y_min), (x_min, y_min), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.line(image, (x_min, y_min), (x_max, y_max), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.line(image, (x_max, y_min), (x_max, y_max), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.line(image, (x_max, y_max), (x_min, y_max), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.line(image, (x_min, y_min), (x_min, y_max), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.circle(image, (x_max, y_max), 2, (10, 10, 220), 3)
    cv2.circle(image, (x_min, y_min), 2, (10, 10, 220), 3)

    return percent_diff, right_up_corner


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

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        points_xyz = []
        for mark in hand_landmarks.landmark:
            points_xyz.extend([mark.x, mark.y])

        points_xyz = np.array(points_xyz).reshape(1, -1)
        difference, right_up_corner = draw_bb(points_xyz)

        pred = clf.predict_proba(points_xyz)[0]
        show_predict(pred, image, right_up_corner)

        if isRecording:
            if np.max(pred) > 0.99 and difference < 0.75 and frame_count % 2 == 0:
                if len(prev_predictions) > 10:
                    letter = mode(prev_predictions)[0][0]
                    if len(predictions) == 0 or predictions[-1] != letter:
                        predictions.append(letter)
                    prev_predictions = []
                else:
                    prev_predictions.append(classes[np.argmax(pred)])
    cv2.imshow(WINDOW, image)

    key = cv2.waitKey(5)

    if key & 0xFF == 27 or key == ord('q'):
        break

    # if "s" is pressed, the predicted letters are written to the list,
    # and then the closest word to the word in the list is calculated using the Levenshtein distance.
    if key == ord('s'):
        if isRecording:
            isRecording = False
            predictions = "".join(list(uj(predictions)))
            print(f'Input: {predictions}, Predict: {min_distance(predictions)}')
        else:
            print('Start')
            predictions = []
            isRecording = True

    # if writer is not None:
    #     writer.write(image)


# if writer is not None:
#     writer.release()

hand.close()
cap.release()
