import csv
import pandas as pd
import os


csv_files = sorted(os.listdir('../dataset/letters'))
SHAPE = (5000, 64)
has_errors = False


def check_length():
    for file in csv_files:
        with open(f'../dataset/letters/{file}', 'r') as csv_file:
            reader = csv.reader(csv_file)
            for idx, row in enumerate(reader):
                if len(row) != 64:
                    global has_errors
                    has_errors = True
                    print(f'Wrond len in: {file} -> len: {len(row)} -> line: {idx + 1}')
    if not has_errors:
        print('(OK)')


def check_shape():
    for file in csv_files:
        df = pd.read_csv(f'../dataset/letters/{file}', header=None)
        if df.shape != SHAPE:
            global has_errors
            has_errors = True
            print(f'Wrong shape in: {file} -> {df.shape}')
    if not has_errors:
        print('(OK)')


def validate():
    print('CHECK LEN', end=' ')
    check_length()
    print('CHECK SHAPE', end=' ')
    check_shape()
    return has_errors
