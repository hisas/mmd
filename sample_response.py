import pandas as pd
from numpy import random

with open('data/sample_response.txt') as f:
    text = f.read().rstrip('\n')

df = pd.read_csv('data/response_test_1.csv', usecols=['Response'])
for _ in range(10):
    print(text + ','.join(random.choice(df['Response'].unique(), 9, replace=False)))
