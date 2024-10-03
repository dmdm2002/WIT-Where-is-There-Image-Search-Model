import os
import re

import pandas as pd
import numpy as np
import shutil


root = 'D:/Side/대회/2024 관광데이터 활용 공모전/DB'

df = pd.read_excel(f'{root}/image_and_keyword.xlsx')

df_arr = np.array(df)
columns = df.columns

for i in range(len(df_arr)):
    name = df_arr[i][0].split('_')[0]
    df_arr[i][0] = name
    print(name)

df_result = pd.DataFrame(df_arr, columns=columns)

print(df_result)

df_result.to_csv(f'{root}/image_and_keyword_label_cleaning.csv', index=False, encoding='utf-8')
