import argparse

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_sim_place_name_based(df, sim_matrix, place_name, top_k=10):
    # 입력한 장소의 index
    place_ = df[df['label'] == place_name]
    place_index_ = place_.index.values

    # 입력한 장소의 유사도 데이터 프레임 추가
    df['similarity'] = sim_matrix[place_name].values

    # 유사도 내림차순 정렬 후 상위 index 추출
    temp = df.sort_values(by='similarity', ascending=False)
    final_index = temp.index.values[:top_k]

    return df.iloc[final_index]

def recommender_place_name_based(tour_places, place_name, top_k=10):
    tour_info_df.drop_duplicates(['label'], inplace=True)  # 중복 제거
    tour_info_df.reset_index(inplace=True)

    tour_places_value_df = tour_places.set_index(tour_places['label'])
    tour_places_value_df = tour_places_value_df.drop(['label'], axis=1)

    cossim = cosine_similarity(tour_places_value_df.values, tour_places_value_df.values)
    sim_matrix = pd.DataFrame(cossim, index=tour_places.label, columns=tour_places.label)

    top_k_df = find_sim_place_name_based(tour_info_df, sim_matrix, place_name, top_k=top_k)

    return top_k_df['label'].values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_place_name',
                        type=str,
                        default='한강시민공원 뚝섬지구(뚝섬한강공원)',
                        help="검색한 장소, 해당 장소를 기반으로 해당 지역과 유사도가 높은 지역을 추천")
    parser.add_argument('--search_user_keyword',
                        type=list,
                        default=[],
                        help="사용자가 선택한 선호하는 관광지의 키워드, 해당 키워드와 유사한 키워드를 가진 지역을 추천")
    parser.add_argument('--top_k',
                        type=int,
                        default=30,
                        help="추출하고 싶은 유사도가 높은 지역의 갯수")
    parser.add_argument('--search_criteria',
                        type=str,
                        default='place',
                        help='검색하고자 하는 기준, place와 keyword 둘중에 하나를 선택하여 사용')
    args = parser.parse_args()

    assert args.search_criteria.lower() == 'place' or args.search_criteria.lower() == 'keyword', "search_criteria 는 place 혹은 keyword만 입력값으로 사용할 수 있습니다."

    tour_info_df = pd.read_csv('../image_and_keyword_final_dataset.csv').drop(['path'], axis=1) # csv 호출 후 필요없는 path 칼럼 제거

    if args.search_criteria.lower() == 'place':
        sim_top_k_places = recommender_place_name_based(tour_info_df, args.search_place_name, args.top_k)
        sim_top_k_places = np.delete(sim_top_k_places, 0)
        print(f'--------------------------[검색지: {args.search_place_name}]--------------------------')
        for idx, name in enumerate(sim_top_k_places):
            print(f'{idx+1}: {name}')
