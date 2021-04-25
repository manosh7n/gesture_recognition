import pandas as pd
from Validator import validate
from GlobalVar import classes

includeAug = False


if validate(classes):
    print(f'SAVE DATASET...{"WITH AUG" if includeAug else "WITHOUT AUG"}')
    combined_csv = pd.concat([pd.read_csv(f'../dataset/letters/{f}.csv', header=None).astype('float64').iloc[:5000, :] for f in classes])
    combined_aug_csv = pd.concat([pd.read_csv(f'../dataset/letters/__{f}.csv', header=None).astype('float64').iloc[:750, :] for f in classes])
    if includeAug:
        combined_csv = pd.concat([combined_csv, combined_aug_csv])
        combined_csv.to_csv("../dataset/dataset_aug.csv", index=False, header=False)
    else:
        combined_csv.to_csv("../dataset/dataset.csv", index=False, header=False)
    print('DONE')
