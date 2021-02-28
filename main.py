import pandas as pd

count = 0
with open('./test_log.txt') as file:
    for line in file:
        line = line.strip()
        count += 1
        result = pd.Series(line.split(' '))
        result.to_csv('./baseline_result/%d.predict.txt'%count, index=False, header=False)
