import cv2
import numpy as np
from tracker.hand_tracker import HandTracker
from app import crop_hand, detector, WINDOW, PALM_MODEL_PATH, ANCHORS_PATH, DEVICE_ID


def save_image(img, name):
    cv2.imwrite(f'dataset/n/n_{name}_{np.random.randint(0, 1e3, 1)[0]}.png', img)


def main():
    capture = cv2.VideoCapture(DEVICE_ID)
    cv2.namedWindow(WINDOW)
    capture.set(cv2.CAP_PROP_FPS, 60)
    write_dataset = False
    value_counts = 0
    name = 1
    frame_counts = 0
    while True:
        _, frame = capture.read()
        bbox = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if bbox is not None:
            crop, bbox_pt = crop_hand(bbox, frame)
            if write_dataset and frame_counts%2 == 0:
                save_image(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR), name)
                name += 1
                value_counts += 1
                print(value_counts)
            cv2.polylines(frame, bbox_pt, 1, (0, 255, 0), 2)
            cv2.imshow('Palm', cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR))
            cv2.moveWindow('Palm', 20, 20)

        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == ord('x'):
            write_dataset = not write_dataset
        cv2.imshow(WINDOW, frame)
        frame_counts += 1

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
