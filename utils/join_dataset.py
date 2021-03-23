import pandas as pd
from validator import validate, csv_files


if not validate():
    print('SAVE DATASET...')
    combined_csv = pd.concat([pd.read_csv(f'../dataset/letters/{f}', header=None).astype('float64') for f in csv_files ])
    combined_csv.to_csv("../dataset/dataset.csv", index=False, header=False)
    print('DONE')
