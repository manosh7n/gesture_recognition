import cv2
import numpy as np
from tracker.hand_tracker import HandTracker
from classifier.gesture_classifier import Classifier
from utils import eval_utils
from scipy.stats import mode

WINDOW = "Gesture recognition"
PALM_MODEL_PATH = "models/palm_detection.tflite"
CLASSIFIER_MODEL_PATH = "models/model_mobile11.h5"
ANCHORS_PATH = "models/anchors.csv"
DEVICE_ID = 0
ALPH = ['А', 'Б', 'В', 'Г', 'И', 'К', 'М', 'Н', 'О', 'С', 'У']

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('videos/output_01.avi', fourcc, 9, (640, 480), True)
detector = HandTracker(PALM_MODEL_PATH,
                       ANCHORS_PATH,
                       box_shift=0.2,
                       box_enlarge=1.2)
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
        cv2.putText(frame, f'{sign}: {predict[i]:.2f}',
                    (5, 20 * (i + 1)), cv2.FONT_HERSHEY_COMPLEX, 0.67,
                    (50, 224, 30) if pred_sign == i else (20, 20, 230), 2)


def main():
    capture = cv2.VideoCapture(DEVICE_ID)
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar('Classification ', WINDOW, 1, 1, lambda: None)
    capture.set(cv2.CAP_PROP_FPS, 60)

    evaluate = False
    eval_utils.clear_folder()
    frame_count = 0
    predictions = []
    word = ''
    while True:
        _, frame = capture.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox = detector(image)

        if bbox is not None:
            crop, bbox_pt = crop_hand(bbox, frame)
            if cv2.getTrackbarPos('Classification ', WINDOW):
                frame_count += 1
                predict = classifier(crop)
                predictions.append(np.argmax(predict[0]))
                show_predict(predict[0], frame)

                if evaluate:
                    eval_utils.collect_img(crop, predict[0])
                    eval_utils.get_accuracy('Г')

            cv2.polylines(frame, bbox_pt, 1, (50, 224, 30), 2)
            cv2.imshow('Palm', cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR))
            cv2.moveWindow('Palm', 20, 20)

        # progress bar for classification
        cv2.rectangle(frame, (5, 463), (630, 477), (233, 111, 0), -1)
        cv2.rectangle(frame, (5, 463), (int(630 * frame_count / 30), 477), (111, 233, 0), -1)
        if frame_count == 30:
            mode_predict, val_count = mode(predictions)
            if val_count > frame_count // 2:
                word += ALPH[mode_predict[0]]
            frame_count = 0
            predictions = []

        cv2.rectangle(frame, (5, 440), (630, 460), (250, 250, 250), -1)
        cv2.putText(frame, word, (5, 458), cv2.FONT_HERSHEY_COMPLEX, 0.7, (20, 20, 20), 2)
        key = cv2.waitKey(1)
        if key == 27:
            break
        # evaluate mode
        if key == ord('e'):
            evaluate = not evaluate
        # clear predicted word
        if key == ord('c') or len(word) >= 45:
            word = ''
            frame_count = 0

        cv2.imshow(WINDOW, frame)
        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
