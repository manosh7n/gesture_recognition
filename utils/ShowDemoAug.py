import cv2
import random
import numpy as np
from DrawWrongPredict import get_pairs, draw_hand, h, w


letter = 'Б'

with open(f'../dataset/letters/for_aug/{letter}_for_aug.csv', 'r') as fin:

    for line in fin.readlines():
        line = list(map(float, line.split(',')))

        pairs = get_pairs(line)
        image = np.zeros((h, w, 3), np.uint8)
        draw_hand(image, pairs)

        while True:
            cv2.imshow("Hand", image)
            key = cv2.waitKey(1000)
            if key == ord('q'):
                break
            if key == ord('n'):
                pairs = get_pairs(line, aug=True, alpha=random.uniform(-15, 15), shift=0.02)
                image = np.zeros((h, w, 3), np.uint8)
                draw_hand(image, pairs)
