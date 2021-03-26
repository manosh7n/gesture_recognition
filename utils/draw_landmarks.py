import cv2
import numpy as np
import pandas as pd
from global_var import *

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
               (5, 6), (6, 7), (7, 8), (5, 9),
               (9, 10), (10, 11), (11, 12), (9, 13),
               (13, 14), (14, 15), (15, 16), (13, 17),
               (17, 18), (18, 19), (19, 20), (0, 17)]


def delete_outliers(index, label):
    letter = ALPH[label]
    df_letter = pd.read_csv(f'../dataset/letters/{letter}.csv', header=None)
    print(f'DELETED: ({letter})')
    df_letter.drop(df_letter.index[index % 5000], inplace=True)
    df_letter.to_csv(f'../dataset/letters/{letter}.csv', index=False, header=False)


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


df = pd.read_csv('../dataset/dataset.csv', header=None)

with open('../dataset/wrong_predict.txt', 'r') as fin:
    _exit = False
    remaining = []
    h, w = 650, 700
    lines = fin.readlines()
    for count, prediction in enumerate(lines, 1):
        idx, pred, true = map(int, prediction.split())
        remaining.extend([idx, pred, true])

        if not _exit:
            example = df.iloc[idx, :63].to_numpy()
            pairs = get_pairs(example)
            image = np.zeros((h, w, 3), np.uint8)
            draw_hand(image, pairs)
            print(f'Pred: {ALPH[pred]} | True: {ALPH[true]}  ({count}/{len(lines)})')

            while True:
                cv2.imshow("Hand", image)
                key = cv2.waitKey(1000)
                if key == ord('n'):
                    break
                if key == ord('r'):
                    delete_outliers(idx, true)
                    remaining = remaining[:-3]
                    break
                if key == ord('q'):
                    _exit = True
                    break

with open('../dataset/wrong_predict.txt', 'w') as fout:
    for line in list(zip(remaining[::3], remaining[1::3], remaining[2::3])):
        print(*line, file=fout)
