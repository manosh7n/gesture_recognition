import csv
import pandas as pd
import os


has_errors = False


def check_shape(classes):
    global has_errors
    letters = []
    for file in classes:
        df = pd.read_csv(f'../dataset/letters/{file}.csv', header=None)
        if df.shape[0] < 5000 or df.shape[1] != 43:
            print(f'Wrong shape in: {file} -> {df.shape}')
            has_errors = True
            letters.append(file[0])
        if df.shape[0] > 5000:
            print(f'Oversize in: {file} -> {df.shape[0]}')
    for file in classes:
        df = pd.read_csv(f'../dataset/letters/__{file}.csv', header=None)
        if df.shape[0] < 750 or df.shape[1] != 43:
            print(f'Wrong shape in: {file} -> {df.shape}')
            has_errors = True
            letters.append(file[0])
        if df.shape[0] > 750:
            print(f'Oversize in: {file} -> {df.shape[0]}')

    if not has_errors:
        print('OK')
    else:
        print(f'ERROR in: {letters}')


def validate(classes):
    print('CHECK SHAPE...')
    check_shape(classes)
    return not has_errors


if __name__ == '__main__':
    validate()


