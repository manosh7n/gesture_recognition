import pandas as pd
import numpy as np
import cv2
import csv
from DrawWrongPredict import get_pairs, draw_hand, h, w
from GlobalVar import *


def draw_sample(image):
    sample = df.sample().to_numpy()[0][:-1]
    pairs = get_pairs(example=sample)
    draw_hand(image, pairs)
    return sample


letter = 'Я'

df = pd.read_csv(f'../dataset/letters/{letter}.csv', header=None)
image = np.zeros((h, w, 3), np.float32)
sample = draw_sample(image)
count = 0

with open(f'../dataset/letters/for_aug/{letter}_for_aug.csv', 'a+') as file:
    writer = csv.writer(file)
    while True:
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(1000)
        if key == ord('q') or count == 75:
            break
        if key == ord('a') or key == ord('n'):
            image = np.zeros((h, w, 3), np.float32)
            sample = draw_sample(image)
            if key == ord('a'):
                writer.writerow(sample)
                count += 1
                print(f'RECORDED: {count}')
