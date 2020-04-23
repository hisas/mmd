import pandas as pd
from numpy import random

with open('data/sample_action.txt') as f:
    text = f.read().rstrip('\n')

df = pd.read_csv('data/action_test_1.csv', usecols=['Action'])
for _ in range(10):
    print(text + ','.join(random.choice(df['Action'].unique(), 9, replace=False)))
