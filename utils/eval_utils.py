import cv2
import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score


def clear_folder():
    files = glob.glob('utils/acc_images/*')
    for f in files:
        os.remove(f)
    print('All images from utils/acc_images removed!')


def collect_img(image, predict):
    name = ALPH[np.argmax(predict)]
    cv2.imwrite(f'utils/acc_images/{name}_{np.random.randint(0, 1e5, 1)[0]}.png', image)


def get_accuracy(y_true):
    y_pred = os.listdir('utils/acc_images')
    y_pred = [file[0] for file in y_pred]
    y_true = [y_true] * len(y_pred)
    print(f'acc: {accuracy_score(y_true, y_pred)} len: {len(y_pred)}')
