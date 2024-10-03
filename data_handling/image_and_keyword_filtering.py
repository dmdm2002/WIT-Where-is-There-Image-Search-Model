import pandas as pd
pd.set_option('display.max_columns', 100)
import numpy as np


root = 'D:/Side/대회/2024 관광데이터 활용 공모전/DB'

# 이미지 위치 + 라벨 + 키워드 데이터
image_and_keyword = pd.read_csv(f'{root}/image_and_keyword_label_merge.csv', encoding='utf-8')
# 근린 공원 제거
drop_park = image_and_keyword[image_and_keyword['label'].str.contains('근린공원')].index
image_and_keyword.drop(drop_park, inplace=True)

########################################################################################################################
"""우선 사용 x"""
# # 서울시 관광지 검색순위 데이터
# search_best = pd.read_csv(f'{root}/2023 지역별 관광지 검색순위.csv', encoding='cp949')
#
#
# # 교통시설 제거
# drop_cate = search_best[search_best['소분류 카테고리'] == '교통시설'].index
# search_best.drop(drop_cate, inplace=True)
#
# search_top_100 = search_best[:100]['관광지명']
# print(search_top_100)
########################################################################################################################

# 데이터 양이 10개 이하인 것들은 제거
df_label_count = image_and_keyword['label'].value_counts()
df_labels = image_and_keyword['label']

many_number = ['', 0]
print(df_labels)
drop_under_ten_label_list = []

for label_name in df_labels:
    if many_number[1] <= df_label_count[label_name]:
        many_number[0] = label_name
        many_number[1] = df_label_count[label_name]

    if df_label_count[label_name] <= 9:
        drop_under_ten_label_list.append(label_name)

drop_under_ten_label_list = list(set(drop_under_ten_label_list))
print(f"list(set(arr) : {drop_under_ten_label_list}")
for drop_label_name in drop_under_ten_label_list:
    print(f'drop label: {drop_label_name}...')
    drop_under_ten_label_index_list = image_and_keyword[image_and_keyword['label'] == drop_label_name].index
    image_and_keyword.drop(drop_under_ten_label_index_list, inplace=True)

image_and_keyword.reset_index(inplace=True)
print(image_and_keyword)
print(f"Number of Labels = {len(image_and_keyword['label'].unique())}")

image_and_keyword.to_csv(f'{root}/image_and_keyword_final_dataset.csv', index=False, encoding='utf-8-sig')
