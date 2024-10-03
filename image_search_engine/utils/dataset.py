from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import PIL.Image as Image
import pandas as pd
import torch

from image_search_engine.utils.transforms import transform_handler

label_dict = {'3.1독립선언기념탑': 0, '63스퀘어': 1, '강남': 2, '강서역사문화거리': 3, '개운사(서울)': 4, '건청궁': 5, '경복궁': 6,
              '고래사진관 필름현상소': 7, '관악산 호수공원': 8, '관훈동 민씨 가옥 (구 부마도위박영효가옥)': 9, '광나루한강공원': 10,
              '광화문': 11, '광희문': 12, '국립4.19민주묘지': 13, '극락사(서울)': 14, '길상사(서울)': 15, '나석주의사동상': 16,
              '낙선재': 17, '낙원동 아구찜 거리': 18, '낙원동 포장마차거리': 19, '난지한강공원': 20, '남대문 갈치조림골목': 21,
              '남산 야외식물원': 22, '남산골한옥마을': 23, '남산예장공원': 24, '내원사(서울)': 25, '노들섬': 26, '답십리 고미술상가': 27,
              '답십리 영화의 거리': 28, '대원군별장 (석파정)': 29, '대학로': 30, '덕수궁': 31, '도선사(서울)': 32, '동대문 문구완구거리': 33,
              '동대문디자인플라자(DDP)': 34, '동대문역사문화공원': 35, '롯데몰 김포공항점스카이파크': 36, '롯데월드타워 서울스카이': 37,
              '명동': 38, '미어캣파크': 39, '반공 청년운동 순국 열사 기념비': 40, '반포 서래섬': 41, '반포한강공원': 42, '백인제가옥': 43,
              '보광사 보광선원(서울)': 44, '보라매안전체험관': 45, '부암동': 46, '북촌전망대': 47, '북한산국립공원(서울)': 48, '불암산': 49,
              '브이알존 엑스 코엑스 직영점': 50, '비우당': 51, '사직공원(서울)': 52, '산마루놀이터': 53, '삼성 강남': 54, '서울 경교장': 55,
              '서울 구 러시아공사관': 56, '서울 대한의원': 57, '서울 문묘와 성균관': 58, '서울 삼각지 대구탕 골목': 59, '서울 석촌동 고분군': 60,
              '서울 암사동 유적': 61, '서울 약현성당': 62, '서울 영휘원(순헌황귀비)과 숭인원(이진)': 63, '서울 우정총국': 64,
              '서울 의릉(경종,선의왕후) [유네스코 세계유산]': 65, '서울 정동교회': 66, '서울 풍납동 토성': 67, '서울로 7017': 68,
              '서울새활용플라자': 69, '서울식물원': 70, '세종대왕 동상': 71, '세종로공원': 72, '소림사(서울)': 73, '수도박물관': 74,
              '수락산': 75, '순명비유강원석물': 76, '순정효황후윤씨친가': 77, '승동교회': 78, '쌈지길': 79, '압구정 로데오거리': 80,
              '양화한강공원': 81, '여의도공원': 82, '연산군묘': 83, '연세로': 84, '열린송현녹지광장': 85, '우이동 먹거리마을': 86,
              '유관순동상': 87, '응암동 감자국 거리': 88, '이종석별장': 89, '이촌한강공원': 90, '이태원 앤틱 가구 거리': 91,
              '인사동': 92, '인왕사(서울)': 93, '일자산자연공원': 94, '자생식물학습장': 95, '잠실한강공원': 96, '장수마을': 97,
              '장충단공원': 98, '절두산 순교성지': 99, '정법사(서울)': 100, '종로 청계 관광특구': 101, '종로3가 포장마차 거리': 102,
              '종묘 [유네스코 세계유산]': 103, '중랑장미공원': 104, '창경궁': 105, '창경궁대온실': 106, '창덕궁': 107, '창의문(자하문)': 108,
              '천도교중앙대교당': 109, '청계천 버들습지': 110, '청계천': 111, '청량사(서울)': 112, '청와대': 113, '초안산': 114,
              '최순우 옛집': 115, '충무공 이순신 동상': 116, '코코넛박스': 117, '템플스테이 홍보관': 118, '파크하비오 워터킹덤&스파': 119,
              '한강시민공원 뚝섬지구(뚝섬한강공원)': 120, '해풍부원군윤택영댁재실': 121, '흥인지문공원': 122}


class CustomDataset(Dataset):
    def __init__(self, dataset_path, trans):
        super().__init__()
        info = pd.read_csv(dataset_path)
        info['label'] = info['label'].replace(label_dict)

        self.images = info['path'].to_numpy()
        self.labels = info['label'].to_numpy()
        info.drop(['path', 'label'], axis=1, inplace=True)
        self.keywords = info.to_numpy()

        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trans(Image.open(self.images[idx]))
        keyword = torch.tensor(self.keywords[idx], dtype=torch.float32)
        label = self.labels[idx]

        return image, keyword, label


def get_loader(train,
               image_size=224,
               crop=False,
               jitter=False,
               noise=False,
               batch_size=16,
               dataset_path=None):

    if train:
        trans = transform_handler(train=train,
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  noise=noise)

        dataset = CustomDataset(dataset_path, trans)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    else:
        trans = transform_handler(train=False, image_size=image_size)
        dataset = CustomDataset(dataset_path, trans)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return loader

# if __name__ == '__main__':
#     a = CustomDataset('D:/Side/대회/2024 관광데이터 활용 공모전/DB/image_and_keyword_final_dataset.csv')