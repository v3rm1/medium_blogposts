#!/Users/varunravivarma/py3_virtualenv/ml/bin/python3

import pandas as pd

from analysis_core import df_info

TRAIN_DATA = './data/train.csv'
OUT_FILE = './data/analysis/train_analysis.csv'

train_data = pd.read_csv(TRAIN_DATA)

train_analysis = df_info(train_data, OUT_FILE, True)

print(train_analysis[(train_analysis['Percent Nulls'] > 0)]['Column'])
