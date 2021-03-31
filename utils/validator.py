import csv
import pandas as pd
import os


csv_files = sorted(os.listdir('../dataset/letters'))
has_errors = False


def check_length():
    for file in csv_files:
        with open(f'../dataset/letters/{file}', 'r') as csv_file:
            reader = csv.reader(csv_file)
            for idx, row in enumerate(reader):
                if len(row) != 64:
                    global has_errors
                    has_errors = True
                    print(f'Wrong len in: {file} -> len: {len(row)} -> line: {idx + 1}')
    if not has_errors:
        print('OK')


def check_shape():
    letters = []
    for file in csv_files:
        df = pd.read_csv(f'../dataset/letters/{file}', header=None)
        if df.shape[0] < 5000 or df.shape[1] != 64:
            print(f'Wrong shape in: {file} -> {df.shape}')
            global has_errors
            has_errors = True
            letters.append(file[0])
        if df.shape[0] > 5000:
            print(f'Oversize in: {file} -> {df.shape[0]}')

    if not has_errors:
        print('OK')
    else:
        print(f'ERROR in: {letters}')


def validate():
    print('CHECK LEN...')
    check_length()
    print('CHECK SHAPE...')
    check_shape()
    return has_errors


if __name__ == '__main__':
    validate()


