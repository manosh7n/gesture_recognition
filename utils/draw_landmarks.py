import cv2
import numpy as np
import pandas as pd


df = pd.read_csv('../dataset/dataset.csv', header=None)

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
              (5, 6), (6, 7), (7, 8), (5, 9),
              (9, 10), (10, 11), (11, 12), (9, 13),
              (13, 14), (14, 15), (15, 16), (13, 17),
              (17, 18), (18, 19), (19, 20), (0, 17)]
alph = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К',
        'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц',
        'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']


def get_pairs(example):
    points = []
    pairs = []

    for index, point in enumerate(example, 1):
        if index % 3 != 0:
            points.append(point)
    for pair in list(zip(points[::2], points[1::2])):
        pairs.append(pair)

    return pairs


def draw_hand(image, pairs):
    for line in CONNECTIONS:
        cv2.line(image,
                 (int(h * pairs[line[0]][0]), int(w * pairs[line[0]][1])),
                 (int(h * pairs[line[1]][0]), int(w * pairs[line[1]][1])),
                 (0, 155, 0), thickness=1)
    for pair in pairs:
        cv2.circle(image,
                   (int(h * pair[0]), int(w * pair[1])), 1,
                   (0, 0, 155), thickness=4)


while True:
    h, w = 650, 700
    with open('../dataset/wrong_predict.txt', 'r') as fin:

        for prediction in fin.readlines():
            idx, pred, true = map(int, map(float, prediction.split()))
            example = df.iloc[idx, :63].to_numpy()
            pairs = get_pairs(example)

            image = np.zeros((h, w, 3), np.uint8)
            draw_hand(image, pairs)
            print(f'| Pred: {alph[pred]} | True: {alph[true]} |')

            while True:
                cv2.imshow("Hand", image)
                key = cv2.waitKey(1000)
                if key == ord('n'):
                    break
