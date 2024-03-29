import csv
import random
from DrawWrongPredict import get_pairs
from GlobalVar import *

letter = 'А'

with open(f'../dataset/letters/for_aug/{letter}_for_aug.csv', 'r') as fin, \
        open(f'../dataset/letters/__{letter}.csv', 'w') as fout:
    writer = csv.writer(fout)
    for line in fin.readlines():
        line = list(map(float, line.split(',')))

        for _ in range(10):
            pairs = get_pairs(line, aug=True, alpha=random.uniform(-15, 15), shift=0.005)
            sample = []
            for i in pairs:
                sample.extend([i[0], i[1]])
            sample.append(ALPH.index(letter))
            writer.writerow(sample)

