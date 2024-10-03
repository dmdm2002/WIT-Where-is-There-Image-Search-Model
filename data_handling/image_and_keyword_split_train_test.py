import re

import pandas as pd
import numpy as np
np.random.seed(1004)

df = pd.read_csv('D:/Side/대회/2024 관광데이터 활용 공모전/DB/backup/image_and_keyword_final_dataset.csv')
columns = df.columns.to_numpy()
labels = pd.unique(df['label'])

train_test_split_df = pd.DataFrame(columns=columns)
train_info = []
test_info = []
for label in labels:
    label_df = df[df['label'] == label]
    label_df_arr = label_df.to_numpy()

    test_index = np.random.choice(len(label_df_arr)-1, 2)
    for idx, info in enumerate(label_df_arr):
        info[0] = re.compile('D:/Side/TravelAPI/DB/').sub('M:/Side/2024_TourAPI/DB/', info[0])
        if idx in test_index:
            test_info.append(info)
        else:
            train_info.append(info)

train_df = pd.DataFrame(data=train_info, columns=columns)
test_df = pd.DataFrame(data=test_info, columns=columns)

train_df.to_csv('M:/Side/2024_TourAPI/DB/train.csv', encoding='utf-8-sig', index=False)
test_df.to_csv('M:/Side/2024_TourAPI/DB/test.csv', encoding='utf-8-sig', index=False)
