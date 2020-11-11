import cv2
import numpy as np
from tracker.hand_tracker import HandTracker
from classifier.gesture_classifier import Classifier

WINDOW = "Gesture recognition"
PALM_MODEL_PATH = "models/palm_detection.tflite"
CLASSIFIER_MODEL_PATH = "models/model_mobile4_with_new.h5"
ANCHORS_PATH = "models/anchors.csv"
DEVICE_ID = 0
ALPH = ['А', 'Б', 'В', 'Г']

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('videos/output_01.avi', fourcc, 10, (640, 480), True)
detector = HandTracker(PALM_MODEL_PATH,
                       ANCHORS_PATH,
                       box_shift=0.1,
                       box_enlarge=1.1)
classifier = Classifier(CLASSIFIER_MODEL_PATH)


def crop_hand(bbox, frame):
    bbox_pt = np.array(bbox, dtype=np.int)
    bbox_pt = np.reshape(bbox_pt, (-1, 4, 2))
    rect = cv2.boundingRect(bbox_pt)
    x, y, w, h = np.abs(rect)
    crop = frame[y:y + h, x:x + w].copy()
    return crop, bbox_pt


def show_predict(predict, frame):
    pred_sign = np.argmax(predict)
    for i, sign in enumerate(ALPH):
        cv2.putText(frame, f'{sign}: {predict[i]:.3f}',
                    (20, 30*(i+1)), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 255, 0) if pred_sign == i else (0, 0, 255), 2)


def main():
    capture = cv2.VideoCapture(DEVICE_ID)
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar('Classification ', WINDOW, 1, 1, lambda: None)
    capture.set(cv2.CAP_PROP_FPS, 60)

    while True:
        _, frame = capture.read()
        bbox = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if bbox is not None:
            crop, bbox_pt = crop_hand(bbox, frame)
            if cv2.getTrackbarPos('Classification ', WINDOW):
                predict = classifier(crop)
                show_predict(predict[0], frame)

            cv2.polylines(frame, bbox_pt, 1, (0, 255, 0), 2)
            cv2.imshow('Palm', cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR))
            cv2.moveWindow('Palm', 20, 20)

        key = cv2.waitKey(1)
        if key == 27:
            break

        cv2.imshow(WINDOW, frame)
        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
