import io
import base64

import onnxruntime
import cv2
import torch
import numpy as np
import PIL.Image as Image
from torchvision import transforms
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

class Inference:
    def __init__(self):
        super().__init__()
        self.label_dict = {0: '3.1독립선언기념탑', 1: '63스퀘어', 2: '강남', 3: '강서역사문화거리', 4: '개운사(서울)', 5: '건청궁',
                           6: '경복궁', 7: '고래사진관 필름현상소', 8: '관악산 호수공원', 9: '관훈동 민씨 가옥 (구 부마도위박영효가옥)',
                           10: '광나루한강공원', 11: '광화문', 12: '광희문', 13: '국립4.19민주묘지', 14: '극락사(서울)', 15: '길상사(서울)',
                           16: '나석주의사동상', 17: '낙선재', 18: '낙원동 아구찜 거리', 19: '낙원동 포장마차거리', 20: '난지한강공원',
                           21: '남대문 갈치조림골목', 22: '남산 야외식물원', 23: '남산골한옥마을', 24: '남산예장공원', 25: '내원사(서울)',
                           26: '노들섬', 27: '답십리 고미술상가', 28: '답십리 영화의 거리', 29: '대원군별장 (석파정)', 30: '대학로',
                           31: '덕수궁', 32: '도선사(서울)', 33: '동대문 문구완구거리', 34: '동대문디자인플라자(DDP)', 35: '동대문역사문화공원',
                           36: '롯데몰 김포공항점스카이파크', 37: '롯데월드타워 서울스카이', 38: '명동', 39: '미어캣파크', 40: '반공 청년운동 순국 열사 기념비',
                           41: '반포 서래섬', 42: '반포한강공원', 43: '백인제가옥', 44: '보광사 보광선원(서울)', 45: '보라매안전체험관',
                           46: '부암동', 47: '북촌전망대', 48: '북한산국립공원(서울)', 49: '불암산', 50: '브이알존 엑스 코엑스 직영점',
                           51: '비우당', 52: '사직공원(서울)', 53: '산마루놀이터', 54: '삼성 강남', 55: '서울 경교장',
                           56: '서울 구 러시아공사관', 57: '서울 대한의원', 58: '서울 문묘와 성균관', 59: '서울 삼각지 대구탕 골목', 60: '서울 석촌동 고분군',
                           61: '서울 암사동 유적', 62: '서울 약현성당', 63: '서울 영휘원(순헌황귀비)과 숭인원(이진)', 64: '서울 우정총국', 65: '서울 의릉(경종,선의왕후) [유네스코 세계유산]',
                           66: '서울 정동교회', 67: '서울 풍납동 토성', 68: '서울로 7017', 69: '서울새활용플라자', 70: '서울식물원',
                           71: '세종대왕 동상', 72: '세종로공원', 73: '소림사(서울)', 74: '수도박물관', 75: '수락산',
                           76: '순명비유강원석물', 77: '순정효황후윤씨친가', 78: '승동교회', 79: '쌈지길', 80: '압구정 로데오거리',
                           81: '양화한강공원', 82: '여의도공원', 83: '연산군묘', 84: '연세로', 85: '열린송현녹지광장',
                           86: '우이동 먹거리마을', 87: '유관순동상', 88: '응암동 감자국 거리', 89: '이종석별장', 90: '이촌한강공원',
                           91: '이태원 앤틱 가구 거리', 92: '인사동', 93: '인왕사(서울)', 94: '일자산자연공원', 95: '자생식물학습장',
                           96: '잠실한강공원', 97: '장수마을', 98: '장충단공원', 99: '절두산 순교성지', 100: '정법사(서울)',
                           101: '종로 청계 관광특구', 102: '종로3가 포장마차 거리', 103: '종묘 [유네스코 세계유산]', 104: '중랑장미공원',
                           105: '창경궁', 106: '창경궁대온실', 107: '창덕궁', 108: '창의문(자하문)', 109: '천도교중앙대교당', 110: '청계천 버들습지',
                           111: '청계천', 112: '청량사(서울)', 113: '청와대', 114: '초안산', 115: '최순우 옛집', 116: '충무공 이순신 동상', 117: '코코넛박스',
                           118: '템플스테이 홍보관', 119: '파크하비오 워터킹덤&스파', 120: '한강시민공원 뚝섬지구(뚝섬한강공원)', 121: '해풍부원군윤택영댁재실', 122: '흥인지문공원'}        # Add your full label_dict here.

    def base64_2_image(self, base64_string):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(base64_string)))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.jpg_compress(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = Image.fromarray(image)
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 string or image processing error: {e}")

    def jpg_compress(self, image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, image = cv2.imencode('.jpg', image, encode_param)
        if not result:
            raise RuntimeError('Could not encode image!')
        return image

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def searching(self, base64_string, keyword=None, top_k=10):
        img_trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = img_trans(self.base64_2_image(base64_string))
        if keyword is None:
            keyword = [0 for _ in range(17)]
        keyword = torch.tensor(keyword, dtype=torch.float32)
        ort_session = onnxruntime.InferenceSession('./model_weight/ImageSearchModel.onnx', providers=['CPUExecutionProvider'])
        ort_inputs = {
            ort_session.get_inputs()[0].name: self.to_numpy(image.unsqueeze(0)),
            ort_session.get_inputs()[1].name: self.to_numpy(keyword.unsqueeze(0))
        }
        output_name = [output.name for output in ort_session.get_outputs()]
        logits = ort_session.run(output_name, ort_inputs)
        top_ten = np.argsort(logits[0], 1)[0][:top_k]
        top_ten_place = [self.label_dict[label] for label in top_ten]
        return top_ten_place

app = FastAPI()

@app.get("/", response_class=JSONResponse)
async def get_result(base64_string: str, keyword: str = Query(None, description="Keyword for search, as a comma-separated list", example="1,0,0,...")):
    try:
        # Convert the keyword string to a list of floats
        keyword_list = [float(k) for k in keyword.split(',')] if keyword else None
    except ValueError:
        raise HTTPException(status_code=400, detail="Keyword must be a comma-separated list of numbers.")
    
    inf = Inference()
    try:
        result = inf.searching(base64_string, keyword_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return JSONResponse(content={"results": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
