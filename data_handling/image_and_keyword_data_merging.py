import pandas as pd
import numpy as np


root = 'D:/Side/대회/2024 관광데이터 활용 공모전/DB'
image_and_keyword = pd.read_csv(f'{root}/image_and_keyword_label_cleaning.csv', encoding='utf-8')
dataset = pd.read_csv(f'{root}/dataset_info.csv')

dataset_cols = dataset.columns
image_and_keyword_cols = image_and_keyword.columns

cols = []
cols += list(dataset_cols[:2])
cols += list(image_and_keyword_cols[1:])
print(cols)

new_df = pd.DataFrame(columns=cols)
for label_name in image_and_keyword['이미지 이름']:
    keyword_info = np.array(image_and_keyword[image_and_keyword['이미지 이름'] == label_name]).tolist()[0][1:]

    if label_name in list(dataset['label']):
        temp_rows = np.array(dataset[dataset['label'] == label_name]).tolist()
        for row in temp_rows:
            new_row = row[:2] + keyword_info
            temp_df = pd.DataFrame([new_row], columns=cols)
            new_df = pd.concat([new_df, temp_df])

new_df.to_csv(f'{root}/image_and_keyword_label_merge.csv', index=False, encoding='utf-8-sig')
