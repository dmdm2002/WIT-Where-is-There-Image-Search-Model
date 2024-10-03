import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

"""
Content based Filtering 의 기본 로직
1. 특정 관광지의 이름이 들어오면 관광지의 이름을 가지고 있는 index를 뽑아낸다.
2. 코사인 유사도 중 관광지 이름 인덱스에 해당하는 값에서 추천 개수만큼 뽑아낸다.
3. 본인은 제외
4. score 기반으로 정렬
"""
tour_places = pd.read_csv('../image_and_keyword_final_dataset.csv').drop(['path'], axis=1)
tour_places.drop_duplicates(['label'], inplace=True)

tour_places_temp = tour_places.set_index(tour_places['label'])
tour_places_temp = tour_places_temp.drop(['label'], axis=1)

cossim = cosine_similarity(tour_places_temp.values, tour_places_temp.values)
print(cossim)
sim_matrix = pd.DataFrame(cossim, index=tour_places.label, columns=tour_places.label)
print(sim_matrix)


def find_sim_place_name_based(df, sim_matrix, place_name, top_n=10):
    # 입력한 장소의 index
    place_ = df[df['label'] == place_name]
    place_index_ = place_.index.values

    # 입력한 장소의 유도 데이터 프레임 추가
    df['similarity'] = sim_matrix[place_index_, :].reshape(-1, 1)

    # 유사도 내림차순 정렬 후 상위 index 추출
    temp = df.sort_values(by='similarity', ascending=False)
    final_index = temp.index.values[:top_n]

    return df.iloc[final_index]


# def fine_sim_place_user_keyword_based(df, )


