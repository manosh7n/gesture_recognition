import cv2
import numpy as np
from tracker.hand_tracker import HandTracker
from classifier.gesture_classifier import Classifier

WINDOW = "Gesture recognition"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
CLASSIFIER_MODEL_PATH = "models/model.h5"
ANCHORS_PATH = "models/anchors.csv"
DEVICE_ID = 0

BB_COLOR = (0, 255, 0)
THICKNESS = 2

ALPH = ['а', 'в', 'г']

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('videos/output_01.avi', fourcc, 10, (640, 480), True)

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.1,
    box_enlarge=1.1
)
classifier = Classifier(CLASSIFIER_MODEL_PATH)


def crop_hand(bbox, frame):
    bbox_pt = np.array(bbox, dtype=np.int)
    bbox_pt = np.reshape(bbox_pt, (-1, 4, 2))
    rect = cv2.boundingRect(bbox_pt)
    x, y, w, h = np.abs(rect)
    cv2.polylines(frame, bbox_pt, 1, BB_COLOR, THICKNESS)
    crop = frame[y:y + h, x:x + w].copy()
    return crop


def show_predict(predict, frame):
    pred = sorted(list(zip(np.around(predict, 3)[0], ALPH)))
    cv2.putText(frame, f'{pred[2]}',
                (20, 20), cv2.FONT_HERSHEY_COMPLEX,
                0.8, BB_COLOR, THICKNESS)
    cv2.putText(frame, f'{pred[1]}',
                (20, 50), cv2.FONT_HERSHEY_COMPLEX,
                0.7, (0, 0, 130), THICKNESS)
    cv2.putText(frame, f'{pred[0]}',
                (20, 80), cv2.FONT_HERSHEY_COMPLEX,
                0.6, (0, 0, 130), THICKNESS)


def main():
    capture = cv2.VideoCapture(DEVICE_ID)
    cv2.namedWindow(WINDOW)
    capture.set(cv2.CAP_PROP_FPS, 60)

    while True:
        _, frame = capture.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, bbox = detector(image)

        if bbox is not None:
            crop = crop_hand(bbox, frame)
            predict = classifier(crop)
            show_predict(predict, frame)
            # cv2.imshow('Crop', cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR))

        cv2.imshow(WINDOW, frame)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    if writer is not None:
        writer.release()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
